"""GPU preflight for the frozen SiT-S/2 plus CNN VA-Diffusion addon."""

import argparse
import glob
import json
from pathlib import Path
import tempfile
import time

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms

from models import SiT_models
from models_va_diffusion import LATENT_DIM, TOKEN_GRID, VADiffusionSiT
from sample_va_diffusion import generate_sample_set
from train_va_diffusion import ImageNetParquet, LATENT_SCALE, save_training_checkpoint


def training_step(model, optimizer, vae, images, labels):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = vae.encode(images).latent_dist.sample() * LATENT_SCALE
    noise = torch.randn_like(clean)
    t = torch.rand(clean.shape[0], device=clean.device)
    t_image = t[:, None, None, None]
    x_t = t_image * clean + (1.0 - t_image) * noise
    target_velocity = clean - noise

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, kl_by_level = model(x_t, t, labels, clean_latent=clean)
        flow = F.mse_loss(prediction.float(), target_velocity.float())
        kl = torch.stack([
            kl_by_level[level].mean()
            for level in ("e4", "e8", "e16")
        ]).mean()
        loss = flow + 0.03 * kl
    loss.backward()
    optimizer.step()
    return {
        "clean": clean,
        "x_t": x_t,
        "t": t,
        "prediction": prediction,
        "kl_by_level": kl_by_level,
        "loss": loss.detach(),
    }


def main(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU index {args.gpu} is unavailable")
    if not args.base_ckpt.is_file():
        raise FileNotFoundError(args.base_ckpt)
    if not args.vae.is_dir():
        raise FileNotFoundError(args.vae)
    data_files = [Path(path) for path in sorted(glob.glob(args.data_glob))]
    if not data_files:
        raise FileNotFoundError(f"no parquet files matched: {args.data_glob}")

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    base = SiT_models["SiT-S/2"](input_size=32)
    base.load_state_dict(load_file(str(args.base_ckpt), device="cpu"), strict=True)
    model = VADiffusionSiT(base).to(device).train()

    counts = model.addon_parameter_counts()
    addon_total = sum(counts.values())
    if not 95_000_000 <= addon_total <= 110_000_000:
        raise RuntimeError(f"addon parameter count is outside 100M scale: {addon_total:,}")
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("frozen SiT base contains trainable parameters")
    if any(torch.count_nonzero(adapter.project.weight) for adapter in model.adapters.values()):
        raise RuntimeError("ZeroProject is not zero initialized")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
    ])
    dataset = ImageNetParquet(
        data_files,
        transform,
        seed=0,
        shuffle_buffer=1000,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=True,
        generator=torch.Generator().manual_seed(0),
    )
    vae = AutoencoderKL.from_pretrained(str(args.vae)).eval().requires_grad_(False).to(device)
    optimizer = torch.optim.AdamW(model.addon_parameters(), lr=1e-4, weight_decay=0.0)
    loader_iterator = iter(loader)
    images, labels = next(loader_iterator)
    images = images.to(device)
    labels = labels.to(device)
    first_step = training_step(model, optimizer, vae, images, labels)

    base_grad_tensors = sum(parameter.grad is not None for parameter in model.base.parameters())
    addon_grad_tensors = sum(parameter.grad is not None for parameter in model.addon_parameters())
    if base_grad_tensors != 0:
        raise RuntimeError("frozen SiT base received gradients")
    if addon_grad_tensors == 0:
        raise RuntimeError("addon received no gradients")
    if first_step["prediction"].shape != first_step["x_t"].shape:
        raise RuntimeError(f"prediction shape mismatch: {tuple(first_step['prediction'].shape)}")

    checkpoint_bytes = 0
    checkpoint_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="va_diffusion_verify_") as temporary_dir:
        checkpoint_path = Path(temporary_dir) / "latest.pt"
        start = time.monotonic()
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=0,
            batch_in_epoch=1,
            global_step=1,
            args=args,
        )
        checkpoint_seconds = time.monotonic() - start
        checkpoint_bytes = checkpoint_path.stat().st_size
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (state["epoch"], state["batch_in_epoch"], state["global_step"]) != (0, 1, 1):
            raise RuntimeError("checkpoint position metadata did not round-trip")

        next_images, next_labels = next(loader_iterator)
        next_images = next_images.to(device)
        next_labels = next_labels.to(device)
        expected_step = training_step(model, optimizer, vae, next_images, next_labels)
        reference_parameter = next(model.addon_parameters())
        expected_parameter = reference_parameter.detach().clone()

        model.load_addon_state_dict(state["addon"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_state"]["cpu"])
        torch.cuda.set_rng_state(state["rng_state"]["cuda"])

        resumed_iterator = iter(loader)
        next(resumed_iterator)
        resumed_images, resumed_labels = next(resumed_iterator)
        resumed_images = resumed_images.to(device)
        resumed_labels = resumed_labels.to(device)
        if not torch.equal(next_images, resumed_images) or not torch.equal(next_labels, resumed_labels):
            raise RuntimeError("resumed data position does not match uninterrupted training")
        resumed_step = training_step(model, optimizer, vae, resumed_images, resumed_labels)
        torch.testing.assert_close(reference_parameter, expected_parameter, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(resumed_step["loss"], expected_step["loss"], rtol=1e-6, atol=1e-7)

    prior_noises = {
        level: torch.randn(
            args.batch_size,
            LATENT_DIM,
            TOKEN_GRID,
            TOKEN_GRID,
            device=device,
        )
        for level in ("e4", "e8", "e16")
    }
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prior_prediction, _ = model(
            resumed_step["x_t"],
            resumed_step["t"],
            resumed_labels,
            prior_noises=prior_noises,
        )
    if prior_prediction.shape != resumed_step["x_t"].shape:
        raise RuntimeError(f"prior prediction shape mismatch: {tuple(prior_prediction.shape)}")

    with tempfile.TemporaryDirectory(prefix="va_diffusion_samples_") as sample_dir:
        metadata_path, sample_files = generate_sample_set(
            model,
            vae,
            Path(sample_dir),
            global_step=2,
            labels=(207, 360, 387, 974),
            cfg_scale=1.5,
            nfes=(1, 3, 5, 10),
            temperatures=(0.0, 0.3, 0.6),
            seed=0,
            bf16=True,
        )
        metadata = json.loads(metadata_path.read_text())
        if len(sample_files) != 48 or len(metadata["files"]) != 48:
            raise RuntimeError("Euler sample grid must contain exactly 48 images")
        if not all((Path(sample_dir) / name).is_file() for name in sample_files):
            raise RuntimeError("Euler sample grid contains missing files")

    result = {
        "device": f"cuda:{args.gpu}",
        "gpu": torch.cuda.get_device_name(device),
        "addon_parameters": addon_total,
        "addon_parameter_counts": counts,
        "frozen_base_parameters": sum(p.numel() for p in model.base.parameters()),
        "prediction_shape": list(first_step["prediction"].shape),
        "clean_latent_shape": list(first_step["clean"].shape),
        "dataset_shards": len(data_files),
        "kl_levels": sorted(first_step["kl_by_level"]),
        "base_grad_tensors": base_grad_tensors,
        "addon_grad_tensors": addon_grad_tensors,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "checkpoint_mib": round(checkpoint_bytes / 1024**2, 1),
        "checkpoint_seconds": round(checkpoint_seconds, 2),
        "checkpoint_exact_next_step": "passed",
        "euler_sample_files": len(sample_files),
    }
    print(json.dumps(result, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=Path(
            "/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/"
            "transformer/diffusion_pytorch_model.safetensors"
        ),
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--data-glob",
        default="/home/scpark/data/imagenet-1k-adm-crop-256/data/train-*.parquet",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=Path("/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/vae"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
