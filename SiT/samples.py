"""Publish a complete, fixed-noise Euler comparison for one checkpoint."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from experiment import CLASSES, STEPS, TEMPS, grid_name, read_json, write_json
from model import LEVELS

SCALE = 0.18215


def complete(run, step):
    folder = run / "samples" / str(step)
    manifest = read_json(folder / "manifest.json", {})
    names = [grid_name(n, t, i) for n in STEPS for t in TEMPS for i in range(len(CLASSES))]
    return (manifest.get("step") == step and manifest.get("files") == names
            and all((folder / name).is_file() for name in names))


@torch.inference_mode()
def publish(run, step, model, vae):
    root = run / "samples"
    root.mkdir(parents=True, exist_ok=True)
    if complete(run, step):
        return
    generator = torch.Generator(device="cuda").manual_seed(0)
    initial = torch.randn(4, 4, 32, 32, device="cuda", generator=generator)
    eps = {key: torch.randn(4, 64, 16, 16, device="cuda", generator=generator) for key in LEVELS}
    labels = torch.tensor((*CLASSES, 1000, 1000, 1000, 1000), device="cuda")
    was_training = model.training
    model.eval()
    try:
        with TemporaryDirectory(dir=root) as tmp:
            output = Path(tmp) / "grid"
            output.mkdir()
            names = []
            hashes = {}
            for nfe in STEPS:
                for temperature in TEMPS:
                    noises = {key: torch.cat((v, v)) * temperature for key, v in eps.items()}
                    x = initial.clone()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for k in range(nfe):
                            t = torch.full((8,), k / nfe, device="cuda")
                            pair, _ = model(torch.cat((x, x)), t, labels, noises=noises)
                            cond, null = pair.float().chunk(2)
                            # Same channel convention as upstream SiT.forward_with_cfg.
                            v = torch.cat((null[:, :3] + 1.5 * (cond[:, :3] - null[:, :3]), cond[:, 3:]), 1)
                            x = x + v / nfe
                        images = vae.decode(x / SCALE).sample.float().add(1).mul(0.5).clamp(0, 1)
                    for slot, image in enumerate(images):
                        name = grid_name(nfe, temperature, slot)
                        path = output / name
                        to_pil_image(image.cpu()).save(path)
                        with Image.open(path) as png:
                            if png.format != "PNG" or png.size != (256, 256):
                                raise ValueError("invalid generated image")
                            png.verify()
                        if step == 0:
                            digest = sha256(path.read_bytes()).digest()
                            if hashes.setdefault((nfe, slot), digest) != digest:
                                raise ValueError("nonzero addon effect at initialization")
                        names.append(name)
            write_json(output / "manifest.json", dict(step=step, files=names, classes=CLASSES, nfe=STEPS,
                       temperatures=TEMPS, cfg=1.5, seed=0, sampler="Euler"))
            output.rename(root / str(step))
    finally:
        model.train(was_training)
