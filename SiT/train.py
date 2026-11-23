"""VA-Diffusion training and its restart verification on a single visible GPU."""
import argparse
from hashlib import sha256
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time

from datasets import Image as HFImage, load_dataset
from diffusers import AutoencoderKL
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from experiment import BATCH, INTERVAL, LR, RUN, beta, read_json, write_json
from model import BASE_FILE, LEVELS, VASiT, WEIGHTS
from samples import SCALE, complete, publish

DATA = Path("/home/scpark/data/imagenet-1k-adm-crop-256/data")
GPU = "GPU-df10b232-92c0-d60e-e2e1-9db16f3032bc"


class TrainingImages(IterableDataset):
    def __init__(self, files):
        self.dataset = load_dataset("parquet", data_files={"train": [str(p) for p in files]},
                                    split="train", streaming=True).cast_column("image", HFImage(decode=False))
        self.epoch = 0

    def __iter__(self):
        # datasets handles sharding between DataLoader workers.
        for record in self.dataset.shuffle(seed=self.epoch, buffer_size=10000):
            image = record["image"]
            source = BytesIO(image["bytes"]) if image["bytes"] is not None else image["path"]
            with Image.open(source) as im:
                rgb = pil_to_tensor(im.convert("RGB"))
            yield rgb, int(record["label"])


def objective(velocity, clean, xt, t, kl, completed):
    prediction = xt.float() + (1 - t[:, None, None, None]) * velocity.float()
    mse = (prediction - clean).square().mean()
    group = {key: kl[key].float().mean() / clean[0].numel() for key in LEVELS}
    rate = sum(group.values())
    weight = beta(completed)
    loss = mse + weight * rate
    return loss, {"loss": loss, "data_mse": mse, "kl": rate, "beta": weight, **group}


def new_optimizer(model):
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.addon.parameters())
    return torch.optim.AdamW(model.addon.parameters(), lr=LR, weight_decay=0)


def update(model, vae, opt, images, labels, completed):
    opt.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = torch.cat([vae.encode(chunk.float().div(127.5).sub(1)).latent_dist.sample().float()
                           for chunk in images.split(8)]) * SCALE
    noise = torch.randn_like(clean)
    t = torch.rand(len(clean), device=clean.device)
    xt = torch.lerp(noise, clean, t[:, None, None, None])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        velocity, kl = model(xt, t, labels, clean=clean)
    loss, metrics = objective(velocity, clean, xt, t, kl, completed)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training objective")
    loss.backward()
    opt.step()
    return {k: v.detach().item() if torch.is_tensor(v) else v for k, v in metrics.items()}


def save_weights(path, model, state, identity):
    metadata = {"format": "va-sit-addon-v1", "inputs": identity, "position": json.dumps(state),
                "cpu_rng": bytes(torch.get_rng_state().tolist()).hex(),
                "cuda_rng": bytes(torch.cuda.get_rng_state().tolist()).hex()}
    weights = {name: value.detach().cpu().contiguous() for name, value in model.addon.named_parameters()}
    tmp = path.with_suffix(".part")
    try:
        save_file(weights, str(tmp), metadata=metadata)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def restore_weights(path, model, identity):
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta["format"] != "va-sit-addon-v1" or meta["inputs"] != identity:
        raise ValueError("checkpoint does not match this run")
    model.addon.load_state_dict(load_file(str(path)), strict=True)
    for key, setter in (("cpu_rng", torch.set_rng_state), ("cuda_rng", torch.cuda.set_rng_state)):
        setter(torch.tensor(list(bytes.fromhex(meta[key])), dtype=torch.uint8))
    return json.loads(meta["position"])


def live_batch(run, step, images):
    live = run / "live"
    live.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=live) as temp:
        stage = Path(temp) / "batch"
        stage.mkdir()
        for i, image in enumerate(images[:4]):
            to_pil_image(image).save(stage / f"{i}.png")
        target = live / str(step)
        if target.exists():
            shutil.rmtree(target)
        stage.rename(target)
    write_json(live / "manifest.json", {"step": step, "count": min(len(images), 4)})
    for old in sorted((d for d in live.iterdir() if d.is_dir() and d.name.isdigit()),
                      key=lambda d: int(d.name))[:-2]:
        shutil.rmtree(old)


def verify(model, vae, files):
    data = TrainingImages(files[:2])
    it = iter(DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(0)))
    first, second = ([v.cuda() for v in next(it)] for _ in range(2))
    del it
    torch.backends.cudnn.deterministic = True
    x = torch.zeros(2, 4, 32, 32, device="cuda")
    t = torch.tensor([0., 1.], device="cuda")
    kl = {k: torch.full((2,), 4096., device="cuda") for k in LEVELS}
    loss, _ = objective(x, x, torch.ones_like(x)*(1-t[:, None, None, None]), t, kl, 5042)
    assert loss.item() == 3.5 and beta(0) == 1e-4 and beta(5041) < 1 and beta(5042) == 1
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.base(x, t, first[1])
        torch.testing.assert_close(model(x, t, first[1], clean=x)[0], reference, rtol=0, atol=0)
    before = {k: v.detach().cpu() for k,v in model.base.state_dict().items()}
    model.train()
    opt = new_optimizer(model)
    update(model, vae, opt, *first, 0)
    for k,v in model.base.state_dict().items():
        assert torch.equal(before[k], v.cpu()), k
    assert all(p.grad is None for p in model.base.parameters())
    assert all(model.addon[k].inject.weight.count_nonzero() > 0 for k in LEVELS)
    with TemporaryDirectory(prefix="vasit-test-") as tmp:
        path = Path(tmp) / "addon.safetensors"
        pos = {"step": 1, "batch": 1, "epoch": 0}
        save_weights(path, model, pos, "test")
        count = sum(p.numel() for p in model.addon.parameters())
        with safe_open(str(path), framework="pt") as f:
            assert set(f.keys()) == set(dict(model.addon.named_parameters()))
        assert 95_000_000 <= count <= 110_000_000 and 0 < path.stat().st_size-4*count < 1024**2
        # Compare the next update to the intended restart with fresh Adam moments.
        metrics = update(model, vae, new_optimizer(model), *second, 1)
        expected = {k: v.cpu() for k,v in model.addon.state_dict().items()}
        fresh = VASiT().cuda().train()
        assert restore_weights(path, fresh, "test") == pos
        assert update(fresh, vae, new_optimizer(fresh), *second, 1) == metrics
        for k,v in fresh.addon.state_dict().items():
            assert torch.equal(v.cpu(), expected[k]), k
        print(json.dumps({"base_equal":True,"frozen":sum(p.numel() for p in model.base.parameters()),
                          "addon":count,"checkpoint_bytes":path.stat().st_size,"fresh_adam_exact":True}), flush=True)


def train(run, files, model, vae, resume):
    run.mkdir(exist_ok=True, parents=True)
    path = run / "addon.safetensors"
    hist = run / "history.jsonl"
    if not resume and any((run/n).exists() for n in (hist.name, path.name, "status.json")):
        raise FileExistsError("existing training run; explicitly resume it")
    source = BASE_FILE.stat()
    identity = sha256((str(BASE_FILE) + str(source.st_size) + str(source.st_mtime_ns) +
                       "|workers=4,batch=128,seed=0|" + "|".join(p.name + str(p.stat().st_size) for p in files)).encode()).hexdigest()
    state = {"step":0, "batch":0, "epoch":0}
    last = {}
    if resume:
        state = restore_weights(path, model, identity)
        if hist.exists():
            pending = hist.with_suffix(".part")
            with hist.open() as inp, pending.open("w") as out:
                for line in inp:
                    if line.endswith("\n"):
                        point = json.loads(line)
                        if point["step"] <= state["step"]:
                            last = point
                            out.write(line)
            pending.replace(hist)
    opt = new_optimizer(model)
    data = TrainingImages(files)
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(data, batch_size=BATCH, num_workers=4, pin_memory=True, generator=generator)
    last_saved = state["step"] if resume else None
    next_save = time.monotonic() + INTERVAL
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def status(phase, **extra):
        now = time.time()
        write_json(run / "status.json", {**state, "phase":phase, "pid":os.getpid(), "updated":now,
                   "batch_size":BATCH,"lr":LR,"latest":last,"last_checkpoint_step":last_saved,
                   "last_checkpoint_at":path.stat().st_mtime if path.exists() else None,
                   "next_checkpoint_at":now+max(0,next_save-time.monotonic()) if not stopping and phase!="failed" else None,
                   **extra})

    def save_and_sample():
        nonlocal last_saved, next_save
        save_weights(path, model, state, identity)
        last_saved = state["step"]
        next_save = time.monotonic() + INTERVAL
        status("sampling")
        publish(run, state["step"], model, vae)

    try:
        if not complete(run, 0):
            if resume:
                raise ValueError("run has lost its original baseline")
            status("baseline")
            publish(run, 0, model, vae)
        if resume and not complete(run, state["step"]):
            publish(run, state["step"], model, vae)
        status("starting")
        with hist.open("a", buffering=1) as log:
            while not stopping:
                data.epoch = state["epoch"]
                for images, labels in itertools.islice(loader, state["batch"], None):
                    if stopping:
                        break
                    metrics = update(model, vae, opt, images.cuda(non_blocking=True), labels.cuda(non_blocking=True), state["step"])
                    state["step"] += 1
                    state["batch"] += 1
                    step = state["step"]
                    if step == 1 or step % 10 == 0:
                        avg = .9 * last.get("ema",metrics["loss"]) + .1 * metrics["loss"]
                        last = {"step":step, "ema":avg, **metrics}
                        print(json.dumps(last, allow_nan=False), file=log, flush=True)
                        live_batch(run, step, images)
                        status("training")
                        print(json.dumps(last), flush=True)
                    if time.monotonic() >= next_save:
                        save_and_sample()
                        status("training")
                if not stopping:
                    state.update(epoch=state["epoch"]+1, batch=0)
        if state["step"] and last_saved != state["step"]:
            save_and_sample()
        status("stopped")
    except Exception as exc:
        status("failed", error=type(exc).__name__)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RUN)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU or torch.cuda.device_count() != 1:
        raise RuntimeError("this experiment requires only server 67 physical GPU 1")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    files = sorted(DATA.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError("no training shards")
    model = VASiT().cuda().train()
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS / "vae"), local_files_only=True).requires_grad_(False).eval().cuda()
    if args.verify:
        verify(model, vae, files)
    else:
        train(args.run, files, model, vae, args.resume)
