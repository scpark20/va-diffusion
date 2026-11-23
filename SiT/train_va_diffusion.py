"""Addon-only DDP training for the SiT-S/2 instance of VA-Diffusion."""

import argparse
import glob
from io import BytesIO
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models import SiT_models
from models_va_diffusion import VADiffusionSiT
from sample_va_diffusion import generate_sample_set


LATENT_SCALE = 0.18215


class ImageNetParquet(IterableDataset):
    def __init__(self, data_files, transform, seed, shuffle_buffer):
        super().__init__()
        self.data_files = [str(path) for path in data_files]
        self.row_count = sum(parquet.ParquetFile(path).metadata.num_rows for path in self.data_files)
        self.dataset = load_dataset(
            "parquet",
            data_files={"train": self.data_files},
            split="train",
            streaming=True,
        )
        self.transform = transform
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.epoch = 0

    def __len__(self):
        return self.row_count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        dataset = self.dataset.shuffle(seed=self.seed + self.epoch, buffer_size=self.shuffle_buffer)
        worker = get_worker_info()
        if worker is not None:
            dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id)
        for row in dataset:
            encoded = row["image"]
            if isinstance(encoded, dict) and encoded.get("bytes") is not None:
                source = BytesIO(encoded["bytes"])
            elif isinstance(encoded, dict) and encoded.get("path"):
                source = encoded["path"]
            else:
                raise ValueError("ImageNet parquet image must contain bytes or path")
            with Image.open(source) as image_file:
                image = image_file.convert("RGB")
            yield self.transform(image), int(row["label"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_device(gpu_index: int) -> torch.device:
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(f"GPU index {gpu_index} is unavailable")
    torch.cuda.set_device(gpu_index)
    return torch.device("cuda", gpu_index)


def load_frozen_base(checkpoint_path: Path, device: torch.device):
    base = SiT_models["SiT-S/2"](input_size=32)
    state = load_file(str(checkpoint_path), device="cpu")
    base.load_state_dict(state, strict=True)
    if state["final_layer.linear.weight"].count_nonzero() == 0:
        raise ValueError("base checkpoint has an untrained zero output head")
    return base.eval().requires_grad_(False).to(device)


def beta_at_step(step: int, total_steps: int, beta_max: float) -> float:
    warmup_steps = max(1, round(total_steps * 0.30))
    return beta_max * min((step + 1) / warmup_steps, 1.0)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str))
    temporary.replace(path)


@torch.no_grad()
def save_live_batch(images: torch.Tensor, output_dir: Path, count: int = 4) -> None:
    for index, image in enumerate(images[:count]):
        save_image(
            image,
            output_dir / f"live_{index:02d}.png",
            normalize=True,
            value_range=(-1, 1),
        )


def collect_rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    }


def save_training_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    batch_in_epoch,
    global_step,
    args,
):
    checkpoint_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({
        "addon": model.addon_state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "rng_state": collect_rng_state(),
        "args": checkpoint_args,
    }, path)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    data_files = [Path(path) for path in sorted(glob.glob(args.data_glob))]
    if not data_files:
        raise FileNotFoundError(f"no parquet files matched: {args.data_glob}")
    if not args.base_ckpt.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {args.base_ckpt}")
    if not args.vae.is_dir():
        raise FileNotFoundError(f"local VAE directory not found: {args.vae}")

    device = setup_device(args.gpu)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_batch").mkdir(exist_ok=True)
    (args.output_dir / "samples").mkdir(exist_ok=True)

    base = load_frozen_base(args.base_ckpt, device)
    model = VADiffusionSiT(base).to(device)
    if any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("SiT base must remain frozen")

    addon_parameters = list(model.addon_parameters())
    addon_counts = model.addon_parameter_counts()
    addon_total = sum(addon_counts.values())
    if not 95_000_000 <= addon_total <= 110_000_000:
        raise RuntimeError(f"addon must be approximately 100M parameters, got {addon_total:,}")
    optimizer = torch.optim.AdamW(addon_parameters, lr=args.lr, weight_decay=0.0)

    vae = AutoencoderKL.from_pretrained(str(args.vae)).eval().requires_grad_(False).to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
    ])
    dataset = ImageNetParquet(
        data_files,
        transform,
        args.seed,
        args.shuffle_buffer,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    if len(loader) == 0:
        raise ValueError("dataset is smaller than one per-GPU batch")

    module = model
    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        module.load_addon_state_dict(state["addon"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", state.get("next_epoch", 0)))
        resume_batch_in_epoch = int(state.get("batch_in_epoch", 0))
        global_step = int(state["global_step"])
        rng_state = state.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state["cpu"])
            torch.cuda.set_rng_state(rng_state["cuda"])

    total_steps = args.epochs * len(loader)
    history = read_json(args.output_dir / "history.json", {"loss": []}) if args.resume else {"loss": []}
    run_info = {
        "model": "VA-Diffusion/SiT-S/2",
        "parquet_shards": len(data_files),
        "dataset_size": len(dataset),
        "gpu_index": args.gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "per_gpu_batch": args.batch_size,
        "global_batch": args.batch_size,
        "addon_parameters": addon_total,
        "addon_parameter_counts": addon_counts,
        "frozen_parameters": sum(parameter.numel() for parameter in module.base.parameters()),
    }
    print(json.dumps(run_info))
    write_json(args.output_dir / "status.json", {
        **run_info,
        "state": "prepared",
        "epoch": start_epoch,
        "epochs": args.epochs,
        "step": global_step,
    })
    write_json(args.output_dir / "history.json", history)

    model.train()
    last_log_time = time.monotonic()
    last_checkpoint_time = time.monotonic()
    checkpoint_seconds = args.checkpoint_minutes * 60.0
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)

        for batch_index, (images, labels) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch_in_epoch:
                continue
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                clean_latent = vae.encode(images).latent_dist.sample() * LATENT_SCALE

            noise = torch.randn_like(clean_latent)
            t = torch.rand(clean_latent.shape[0], device=device)
            t_image = t[:, None, None, None]
            x_t = t_image * clean_latent + (1.0 - t_image) * noise
            target_velocity = clean_latent - noise

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                prediction, kl_by_level = model(x_t, t, labels, clean_latent=clean_latent)
                flow_loss = F.mse_loss(prediction.float(), target_velocity.float())
                kl_loss = torch.stack([
                    kl_by_level["e4"].mean(),
                    kl_by_level["e8"].mean(),
                    kl_by_level["e16"].mean(),
                ]).mean()
                beta = beta_at_step(global_step, total_steps, args.beta_max)
                loss = flow_loss + beta * kl_loss

            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.log_every == 0:
                metrics = torch.stack([
                    loss,
                    flow_loss,
                    kl_loss,
                    kl_by_level["e4"].mean(),
                    kl_by_level["e8"].mean(),
                    kl_by_level["e16"].mean(),
                ])
                metrics = metrics.detach().cpu().tolist()
                point = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": metrics[0],
                    "flow": metrics[1],
                    "kl": metrics[2],
                    "kl_e4": metrics[3],
                    "kl_e8": metrics[4],
                    "kl_e16": metrics[5],
                    "beta": beta,
                    "steps_per_second": args.log_every / max(time.monotonic() - last_log_time, 1e-9),
                }
                last_log_time = time.monotonic()
                print(json.dumps(point))
                history["loss"].append(point)
                write_json(args.output_dir / "history.json", history)
                write_json(args.output_dir / "status.json", {
                    **run_info,
                    "state": "training",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "step": global_step,
                    "latest": point,
                })
                save_live_batch(images, args.output_dir / "live_batch")

            checkpoint_due = time.monotonic() - last_checkpoint_time >= checkpoint_seconds
            if checkpoint_due:
                write_json(args.output_dir / "status.json", {
                    **run_info,
                    "state": "sampling",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "step": global_step,
                    "latest": history["loss"][-1] if history["loss"] else {},
                })
                save_training_checkpoint(
                    args.output_dir / "latest.pt",
                    module,
                    optimizer,
                    epoch,
                    batch_index + 1,
                    global_step,
                    args,
                )
                generate_sample_set(
                    module,
                    vae,
                    args.output_dir / "samples",
                    global_step,
                    labels=(207, 360, 387, 974),
                    cfg_scale=1.5,
                    nfes=(1, 3, 5, 10),
                    temperatures=(0.0, 0.3, 0.6),
                    seed=args.sample_seed,
                    bf16=args.bf16,
                )
                write_json(args.output_dir / "status.json", {
                    **run_info,
                    "state": "training",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "step": global_step,
                    "latest": history["loss"][-1] if history["loss"] else {},
                    "last_checkpoint_step": global_step,
                })
                last_checkpoint_time = time.monotonic()

        resume_batch_in_epoch = 0

    save_training_checkpoint(
        args.output_dir / "latest.pt",
        module,
        optimizer,
        args.epochs,
        0,
        global_step,
        args,
    )
    generate_sample_set(
        module,
        vae,
        args.output_dir / "samples",
        global_step,
        labels=(207, 360, 387, 974),
        cfg_scale=1.5,
        nfes=(1, 3, 5, 10),
        temperatures=(0.0, 0.3, 0.6),
        seed=args.sample_seed,
        bf16=args.bf16,
    )

    write_json(args.output_dir / "status.json", {
        **run_info,
        "state": "complete",
        "epoch": args.epochs,
        "epochs": args.epochs,
        "step": global_step,
        "latest": history["loss"][-1] if history["loss"] else {},
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-glob",
        default="/home/scpark/data/imagenet-1k-adm-crop-256/data/train-*.parquet",
    )
    parser.add_argument(
        "--base-ckpt",
        type=Path,
        default=Path("/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors"),
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=Path("/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/vae"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32", action="store_false", dest="bf16")
    parser.set_defaults(bf16=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
