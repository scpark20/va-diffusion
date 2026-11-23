"""Continuous single-GPU training for VA-Diffusion / SiT-S/2."""
import argparse
from io import BytesIO
import itertools
import json
from pathlib import Path
import signal
import time

import pyarrow.parquet as pq
import torch
from diffusers import AutoencoderKL
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

from models import SiT_models
from models_va_diffusion import VADiffusionSiT
from sample_va_diffusion import LATENT_SCALE, generate_sample_set
from datasets import load_dataset, Image as DatasetImage

WEIGHTS = Path("/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256")
CKPT = WEIGHTS / "transformer/diffusion_pytorch_model.safetensors"
DATA = Path("/home/scpark/data/imagenet-1k-adm-crop-256/data")


class ImageNetStream(IterableDataset):
    def __init__(self, files, seed):
        self.data = load_dataset("parquet", data_files={"train": list(map(str, files))},
                                 split="train", streaming=True)
        if isinstance(self.data.features["image"], DatasetImage):
            self.data = self.data.cast_column("image", DatasetImage(decode=False))
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        # Hugging Face distributes shards to DataLoader workers on iteration.
        for row in self.data.shuffle(seed=self.seed + self.epoch, buffer_size=10000):
            with Image.open(BytesIO(row["image"]["bytes"])) as image:
                tensor = pil_to_tensor(image.convert("RGB"))
            yield tensor, int(row["label"])


def beta_at_step(step):
    return max(1e-4, min(1.0, (step - 21) / 5021))


def objective(velocity, clean, x_t, t, kl_sums, step):
    pred_clean = x_t.float() + (1 - t[:, None, None, None]) * velocity.float()
    data_mse = F.mse_loss(pred_clean, clean.float())
    rates = {level: raw.float().mean() / clean[0].numel() for level, raw in kl_sums.items()}
    rate = sum(rates.values())
    beta = beta_at_step(step)
    return data_mse + beta * rate, {"data_mse": data_mse, "kl": rate, "beta": beta,
                                    **{f"kl_{level}": value for level, value in rates.items()}}


def make_model(device):
    base = SiT_models["SiT-S/2"](input_size=32)
    base.load_state_dict(load_file(str(CKPT)), strict=True)
    if not base.final_layer.linear.weight.count_nonzero():
        raise ValueError("SiT checkpoint has an untrained output head")
    return VADiffusionSiT(base).to(device)


def update(model, optimizer, vae, images, labels, step):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = (vae.encode(images.float().div(127.5).sub(1)).latent_dist.sample().float() * LATENT_SCALE)
    noise = torch.randn_like(clean)
    t = torch.rand(len(clean), device=clean.device)
    x_t = torch.lerp(noise, clean, t[:, None, None, None])
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        velocity, sums = model(x_t, t, labels, clean_latent=clean)
    loss, values = objective(velocity, clean, x_t, t, sums, step)
    if not loss.detach().isfinite():
        raise FloatingPointError("nonfinite objective before optimizer update")
    loss.backward()
    optimizer.step()
    return {k: float(v.detach()) if torch.is_tensor(v) else v
            for k, v in {"loss": loss, **values}.items()}


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path, model, optimizer, position, settings):
    tmp = path.with_suffix(".pt.tmp")
    try:
        torch.save({"format": 2, "addon": model.addon.state_dict(), "optimizer": optimizer.state_dict(),
                    **position, "settings": settings, "cpu_rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state(next(model.parameters()).device)}, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, settings):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != 2 or state["settings"] != settings:
        raise ValueError("checkpoint format or training settings differ")
    model.load_addon_state_dict(state["addon"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["gpu_rng"], next(model.parameters()).device)
    return {k: state[k] for k in ("epoch", "batch_in_epoch", "global_step")}


def train(args):
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    if not args.resume and any((out / name).exists() for name in ("status.json", "latest.pt", "history.json")):
        raise FileExistsError("existing run: use --resume to continue")
    files = sorted(DATA.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError("no ImageNet training shards")
    settings = {"seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers,
                "files": [p.name for p in files], "base": str(CKPT), "lr": 1e-4}
    dataset = ImageNetStream(files, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = make_model(device).train()
    optimizer = torch.optim.AdamW(model.addon.parameters(), lr=settings["lr"], weight_decay=0)
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS / "vae"), local_files_only=True).eval().requires_grad_(False).to(device)
    pos = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0}
    history = {"loss": []}
    if args.resume:
        pos = restore_checkpoint(latest, model, optimizer, settings)
        if (out / "history.json").is_file():
            history["loss"] = [p for p in json.loads((out / "history.json").read_text())["loss"]
                               if p["step"] <= pos["global_step"]]
    (out / "live_batch").mkdir(exist_ok=True)
    info = {"model": "VA-Diffusion/SiT-S/2", "training_mode": "continuous", "gpu_index": 1,
            "gpu_name": torch.cuda.get_device_name(device), "per_gpu_batch": args.batch_size,
            "parquet_shards": len(files), "dataset_size": sum(pq.ParquetFile(f).metadata.num_rows for f in files),
            "addon_parameters": sum(p.numel() for p in model.addon.parameters()),
            "frozen_parameters": sum(p.numel() for p in model.base.parameters())}

    def status(state, **extra):
        atomic_json(out / "status.json", {**info, **pos, "step": pos["global_step"], "state": state,
                                         "latest": history["loss"][-1] if history["loss"] else {}, **extra})

    def checkpoint():
        save_checkpoint(latest, model, optimizer, pos, settings)
        status("sampling")
        generate_sample_set(model, vae, out / "samples", pos["global_step"], seed=args.seed)

    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status("starting")
    checkpoint_time = time.monotonic() + 30 * 60
    last_saved = None
    try:
        while not stopped:
            dataset.epoch = pos["epoch"]
            batches = itertools.islice(loader, pos["batch_in_epoch"], None)
            for images, labels in batches:
                if stopped:
                    break
                metrics = update(model, optimizer, vae, images.to(device, non_blocking=True),
                                 labels.to(device, non_blocking=True), pos["global_step"])
                pos["global_step"] += 1
                pos["batch_in_epoch"] += 1
                step = pos["global_step"]
                if step == 1 or step % 10 == 0:
                    point = {"step": step, "epoch": pos["epoch"], **metrics}
                    history["loss"].append(point)
                    atomic_json(out / "history.json", history)
                    for i, im in enumerate(images[:4]):
                        target = out / "live_batch" / f"live_{i:02d}.png"
                        tmp = target.with_suffix(".tmp")
                        save_image(im.float().div(255), tmp, format="PNG")
                        tmp.replace(target)
                    status("training")
                    print(json.dumps(point), flush=True)
                if time.monotonic() >= checkpoint_time:
                    checkpoint_time = time.monotonic() + 1800
                    checkpoint()
                    last_saved = step
                    status("training", last_checkpoint_step=step)
            if not stopped:
                pos.update(epoch=pos["epoch"] + 1, batch_in_epoch=0)
        if pos["global_step"] and last_saved != pos["global_step"]:
            checkpoint()
        status("stopped")
    except Exception as exc:
        status("failed", error=type(exc).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())
