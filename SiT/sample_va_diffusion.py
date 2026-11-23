"""Euler sampling for a trained VA-Diffusion SiT-S/2 addon."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from torchvision.utils import save_image

from models import SiT_models
from models_va_diffusion import VADiffusionSiT


LATENT_SCALE = 0.18215


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(base_path: Path, addon_path: Path, device: torch.device):
    base = SiT_models["SiT-S/2"](input_size=32)
    base.load_state_dict(load_file(str(base_path), device="cpu"), strict=True)
    model = VADiffusionSiT(base, latent_dim=16)
    checkpoint = torch.load(addon_path, map_location="cpu")
    model.load_addon_state_dict(checkpoint["addon"])
    return model.eval().requires_grad_(False).to(device), checkpoint


def temperature_name(temperature: float) -> str:
    return f"{temperature:.1f}".replace(".", "p")


@torch.inference_mode()
def euler_sample(
    model,
    initial_noise,
    labels,
    prior_eps,
    nfe,
    temperature,
    cfg_scale,
):
    x = initial_noise.clone()
    scaled_prior = {
        level: epsilon * temperature
        for level, epsilon in prior_eps.items()
    }

    for step in range(nfe):
        t = torch.full(
            (x.shape[0],),
            step / nfe,
            device=x.device,
            dtype=x.dtype,
        )
        if cfg_scale == 1.0:
            velocity = model(x, t, labels, prior_noises=scaled_prior)[0]
        else:
            x_pair = torch.cat([x, x], dim=0)
            t_pair = torch.cat([t, t], dim=0)
            labels_pair = torch.cat([labels, torch.full_like(labels, 1000)], dim=0)
            prior_pair = {
                level: torch.cat([epsilon, epsilon], dim=0)
                for level, epsilon in scaled_prior.items()
            }
            prediction = model(x_pair, t_pair, labels_pair, prior_noises=prior_pair)[0]
            conditional, unconditional = prediction.chunk(2)
            velocity = conditional.clone()
            velocity[:, :3] = unconditional[:, :3] + cfg_scale * (
                conditional[:, :3] - unconditional[:, :3]
            )
        x = x + velocity / nfe
    return x


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if len(args.labels) != 4:
        raise ValueError("exactly four class labels are required")
    for path in (args.base_ckpt, args.addon_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.vae.is_dir():
        raise FileNotFoundError(args.vae)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.base_ckpt, args.addon_ckpt, device)
    vae = AutoencoderKL.from_pretrained(str(args.vae)).eval().requires_grad_(False).to(device)
    labels = torch.tensor(args.labels, device=device, dtype=torch.long)
    initial_noise = torch.randn(4, 4, 32, 32, device=device)
    prior_eps = {
        level: torch.randn(4, 256, 16, device=device)
        for level in ("e4", "e8", "e16")
    }
    epoch = int(checkpoint.get("next_epoch", 0))
    files = []

    for temperature in args.temperatures:
        for nfe in args.nfes:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                latent = euler_sample(
                    model,
                    initial_noise,
                    labels,
                    prior_eps,
                    nfe,
                    temperature,
                    args.cfg_scale,
                )
                images = vae.decode(latent / LATENT_SCALE).sample
            for index, image in enumerate(images):
                name = (
                    f"epoch_{epoch:04d}_nfe_{nfe}_"
                    f"temp_{temperature_name(temperature)}_{index:02d}.png"
                )
                save_image(image, args.output_dir / name, normalize=True, value_range=(-1, 1))
                files.append(name)

    metadata = {
        "model": "VA-Diffusion/SiT-S/2",
        "sampler": "Euler",
        "epoch": epoch,
        "nfes": args.nfes,
        "temperatures": args.temperatures,
        "samples_per_setting": 4,
        "cfg_scale": args.cfg_scale,
        "labels": args.labels,
        "seed": args.seed,
        "temperature_definition": "z = mu_p + temperature * sigma_p * epsilon",
        "shared_initial_noise": True,
        "shared_prior_epsilon": True,
        "base_checkpoint_sha256": sha256(args.base_ckpt),
        "addon_checkpoint_sha256": sha256(args.addon_ckpt),
        "files": files,
    }
    metadata_path = args.output_dir / f"epoch_{epoch:04d}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"generated": len(files), "metadata": str(metadata_path)}))


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--addon-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs=4, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--nfes", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32", action="store_false", dest="bf16")
    parser.set_defaults(bf16=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
