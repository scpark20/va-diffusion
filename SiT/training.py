"""Single-GPU continuous ImageNet training. Only the CNN addon is optimized and saved."""
import argparse
from hashlib import sha256
from io import BytesIO
import itertools
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time

from datasets import Image as DatasetImage, load_dataset
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from addon import VADiffusion, WEIGHTS
from sampling import LATENT_SCALE, generate_grid, load_vae, published_grid

DATA = Path("/home/scpark/data/imagenet-1k-adm-crop-256/data")
BATCH, LR = 128, 1e-4
SAVE_SECONDS = 30 * 60


class ImageNet(IterableDataset):
    def __init__(self, shards):
        self.source = load_dataset("parquet", data_files={"train": list(map(str, shards))},
                                   split="train", streaming=True).cast_column("image", DatasetImage(decode=False))
        self.epoch = 0

    def __iter__(self):
        # The streaming dataset assigns shards to PyTorch loader workers.
        source = self.source.shuffle(seed=self.epoch, buffer_size=10000)
        for row in source:
            value = row["image"]
            with Image.open(BytesIO(value["bytes"]) if value["bytes"] is not None else value["path"]) as im:
                image = pil_to_tensor(im.convert("RGB"))
            yield image, row["label"]


def beta(step):
    return max(1e-4, min(1.0, (step - 21) / 5021))


def loss_terms(velocity, clean, xt, t, sums, step):
    prediction = xt.float() + (1 - t[:, None, None, None]) * velocity.float()
    mse = (prediction - clean.float()).square().mean()
    rates = {f"kl{level}": value.float().mean() / clean[0].numel() for level, value in sums.items()}
    rate = sum(rates.values())
    value = mse + beta(step) * rate
    return value, {"loss": value.detach(), "data_mse": mse.detach(), "kl": rate.detach(),
                   "beta": beta(step), **{k: v.detach() for k, v in rates.items()}}


def optimizer_for(model):
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.addon.parameters())
    return torch.optim.AdamW(model.addon.parameters(), lr=LR, weight_decay=0)


def update(model, vae, optimizer, images, labels, step):
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = torch.cat([vae.encode(chunk.float().div(127.5).sub(1)).latent_dist.sample().float()
                           for chunk in images.split(8)]) * LATENT_SCALE
    noise = torch.randn_like(clean)
    t = torch.rand(len(clean), device=clean.device)
    xt = torch.lerp(noise, clean, t[:, None, None, None])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        velocity, sums = model(xt, t, labels, clean=clean)
    loss, values = loss_terms(velocity, clean, xt, t, sums, step)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training objective")
    loss.backward()
    optimizer.step()
    return {k: float(v) for k, v in values.items()}


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, allow_nan=False))
    temp.replace(path)


def checkpoint(path, model, position, identity):
    device = next(model.parameters()).device
    meta = {"format": "va-diffusion-addon-1", "position": json.dumps(position), "inputs": identity,
            "cpu_rng": bytes(torch.get_rng_state().tolist()).hex(),
            "cuda_rng": bytes(torch.cuda.get_rng_state(device).tolist()).hex()}
    temp = path.with_suffix(".tmp")
    weights = {k: v.detach().cpu().contiguous() for k, v in model.addon.named_parameters()}
    try:
        save_file(weights, str(temp), metadata=meta)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def restore(path, model, identity):
    with safe_open(path, framework="pt") as file:
        meta = file.metadata()
    if meta["format"] != "va-diffusion-addon-1" or meta["inputs"] != identity:
        raise ValueError("checkpoint inputs or format differ")
    model.addon.load_state_dict(load_file(path), strict=True)
    torch.set_rng_state(torch.tensor(list(bytes.fromhex(meta["cpu_rng"])), dtype=torch.uint8))
    torch.cuda.set_rng_state(torch.tensor(list(bytes.fromhex(meta["cuda_rng"])), dtype=torch.uint8),
                             next(model.parameters()).device)
    return json.loads(meta["position"])


def verify(model, vae, data):
    data.epoch = 0
    stream = iter(DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(0)))
    first = [x.cuda() for x in next(stream)]
    second = [x.cuda() for x in next(stream)]
    del stream
    torch.backends.cudnn.deterministic = True
    zeros = torch.zeros(2,4,32,32,device="cuda")
    t = torch.tensor([0.,1.],device="cuda")
    sums = {k: torch.full((2,),4096.,device="cuda") for k in ("4","8","16")}
    value, _ = loss_terms(zeros, zeros, torch.ones_like(zeros)*(1-t[:,None,None,None]), t, sums, 5042)
    assert value.item() == 3.5 and beta(0) == 1e-4
    before = {k: v.cpu().clone() for k, v in model.base.state_dict().items()}
    model.train()
    optimizer = optimizer_for(model)
    update(model, vae, optimizer, *first, 0)
    assert all(p.grad is None for p in model.base.parameters())
    for k, p in model.base.state_dict().items():
        assert torch.equal(before[k], p.cpu()), k
    for k in ("4","8","16"):
        assert model.addon[k].project.weight.count_nonzero() > 0
    pos = {"step":1,"epoch":0,"batch":1}
    with TemporaryDirectory(prefix="va-addon-verify-") as tmp:
        path = Path(tmp)/"addon.safetensors"
        checkpoint(path, model, pos, "verification")
        with safe_open(path, framework="pt") as f:
            assert set(f.keys()) == set(dict(model.addon.named_parameters()))
        count = sum(p.numel() for p in model.addon.parameters())
        assert 95_000_000 <= count <= 110_000_000 and 0 < path.stat().st_size - count * 4 < 1024**2
        # Weights-only resume intentionally starts a fresh AdamW on both branches.
        expected = update(model, vae, optimizer_for(model), *second, 1)
        reference = {k: p.detach().cpu() for k,p in model.addon.named_parameters()}
        fresh = VADiffusion().cuda().train()
        assert restore(path, fresh, "verification") == pos
        actual = update(fresh, vae, optimizer_for(fresh), *second, 1)
        assert expected == actual
        for k,p in fresh.addon.named_parameters():
            assert torch.equal(reference[k], p.cpu()), k
        print(json.dumps({"verified":True, "frozen_elements":sum(p.numel() for p in model.base.parameters()),
                          "addon_elements":count, "checkpoint_bytes":path.stat().st_size,
                          "fresh_adam_next_update_equal": True}), flush=True)


def train(run, resume, do_verify):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    shards = sorted(DATA.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError("no ImageNet training shards")
    data = ImageNet(shards[:2] if do_verify else shards)
    model, vae = VADiffusion().to(device).train(), load_vae(device)
    if do_verify:
        return verify(model, vae, data)
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "addon.safetensors"
    if not resume and any((run/n).exists() for n in ("history.jsonl","status.json",ckpt.name)):
        raise FileExistsError("existing training run; choose --resume")
    identity = sha256(("SiT-S/2|seed=0|workers=4|batch=128|" + str(WEIGHTS) + "|" +
                       "|".join(p.name + ":" + str(p.stat().st_size) for p in shards)).encode()).hexdigest()
    pos = {"step":0,"epoch":0,"batch":0}
    history = run / "history.jsonl"
    ema = None
    latest = {}
    if resume:
        pos = restore(ckpt, model, identity)
        if history.exists():
            temp = history.with_suffix(".tmp")
            with history.open() as src, temp.open("w") as out:
                for line in src:
                    if not line.endswith("\n"):
                        continue
                    record = json.loads(line)
                    if record["step"] <= pos["step"]:
                        latest, ema = record, record["ema"]
                        out.write(line)
            temp.replace(history)
    optimizer = optimizer_for(model)
    loader = DataLoader(data, batch_size=BATCH, num_workers=4, pin_memory=True,
                        generator=torch.Generator().manual_seed(0))
    live = run / "live"
    live.mkdir(exist_ok=True)
    last_save = pos["step"] if resume else None
    next_save = time.monotonic() + SAVE_SECONDS
    stopping = False
    def stop(sig, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(state, **extra):
        now = time.time()
        write_json(run / "status.json", {**pos,"state":state,"latest":latest,"updated_at":now,"batch_size":BATCH,
                   "lr":LR,"last_checkpoint_step":last_save,"last_checkpoint_at":ckpt.stat().st_mtime if ckpt.exists() else None,
                   "next_checkpoint_at":now + max(0, next_save-time.monotonic()) if not stopping and state != "failed" else None,
                   **extra})

    def save_and_sample():
        nonlocal last_save, next_save
        checkpoint(ckpt, model, pos, identity)
        last_save = pos["step"]
        next_save = time.monotonic() + SAVE_SECONDS
        status("sampling")
        generate_grid(model, vae, run, pos["step"])

    try:
        if not published_grid(run,0):
            if resume:
                raise FileNotFoundError("step-zero baseline missing from this run")
            status("sampling")
            generate_grid(model, vae, run, 0)
        if resume and not published_grid(run, pos["step"]):
            generate_grid(model, vae, run, pos["step"])
        status("starting")
        with history.open("a", buffering=1) as log:
            while not stopping:
                data.epoch = pos["epoch"]
                for images, labels in itertools.islice(loader, pos["batch"], None):
                    if stopping:
                        break
                    metrics = update(model, vae, optimizer, images.to(device, non_blocking=True),
                                     labels.to(device, non_blocking=True), pos["step"])
                    pos["step"] += 1
                    pos["batch"] += 1
                    step = pos["step"]
                    if step == 1 or step % 10 == 0:
                        ema = metrics["loss"] if ema is None else .9*ema + .1*metrics["loss"]
                        latest = {"step":step, "ema":ema, **metrics}
                        print(json.dumps(latest, allow_nan=False), file=log, flush=True)
                        for i, im in enumerate(images[:4]):
                            temp = live/f"image{i}.tmp"
                            to_pil_image(im).save(temp, format="PNG")
                            temp.replace(live/f"image{i}.png")
                        write_json(live/"batch.json", {"step":step,"count":min(4,len(images))})
                        status("training")
                        print(json.dumps(latest), flush=True)
                    if time.monotonic() >= next_save:
                        save_and_sample()
                        status("training")
                if not stopping:
                    pos.update(epoch=pos["epoch"]+1, batch=0)
        if pos["step"] and last_save != pos["step"]:
            save_and_sample()
        status("stopped")
    except Exception as exc:
        status("failed", error=type(exc).__name__)
        raise


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run", type=Path, required=True)
    args.add_argument("--resume", action="store_true")
    args.add_argument("--verify", action="store_true")
    options = args.parse_args()
    train(options.run, options.resume, options.verify)
