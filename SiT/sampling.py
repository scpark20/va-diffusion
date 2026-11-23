"""Fixed-seed Euler grids for step zero and every saved addon."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from addon import LEVELS, VADiffusion, WEIGHTS

NFES = (1, 3, 5)
TEMPERATURES = (0.0, 0.3, 0.6)
LABELS = (207, 360, 387, 974)
LATENT_SCALE = 0.18215


def load_vae(device):
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS / "vae"), local_files_only=True)
    return vae.requires_grad_(False).eval().to(device)


def grid_files():
    return [(n, temp, i, f"nfe{n}_t{round(temp * 10):02d}_{i}.png")
            for n in NFES for temp in TEMPERATURES for i in range(4)]


def published_grid(run, step):
    folder = run / "samples" / f"step_{step:08d}"
    try:
        meta = json.loads((folder / "grid.json").read_text())
        return (meta["step"] == step and meta["files"] == [row[-1] for row in grid_files()]
                and meta["labels"] == list(LABELS) and meta["temperatures"] == list(TEMPERATURES)
                and meta["nfes"] == list(NFES) and meta["cfg"] == 1.5 and meta["seed"] == 0
                and all((folder / name).is_file() for name in meta["files"]))
    except (OSError, KeyError, ValueError):
        return False


@torch.inference_mode()
def generate_grid(model, vae, run, step):
    parent = run / "samples"
    parent.mkdir(exist_ok=True, parents=True)
    target = parent / f"step_{step:08d}"
    if target.exists():
        if published_grid(run, step):
            return
        raise ValueError(f"incomplete sample set at {target}")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(0)
    initial = torch.randn((4, 4, 32, 32), generator=generator, device=device)
    eps = {k: torch.randn((4, 64, 16, 16), generator=generator, device=device) for k in LEVELS.values()}
    labels = torch.tensor([*LABELS, 1000, 1000, 1000, 1000], device=device)
    was_training = model.training
    model.eval()
    files = grid_files()
    try:
        with TemporaryDirectory(dir=parent) as tmp:
            stage = Path(tmp) / "grid"
            stage.mkdir()
            for n in NFES:
                for temp in TEMPERATURES:
                    noises = {k: torch.cat([v, v]) * temp for k, v in eps.items()}
                    x = initial.clone()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for i in range(n):
                            t = torch.full((8,), i / n, device=device)
                            pred, _ = model(torch.cat([x, x]), t, labels, noises=noises)
                            cond, uncond = pred.float().chunk(2)
                            velocity = cond.clone()
                            velocity[:, :3] = uncond[:, :3] + 1.5 * (cond[:, :3] - uncond[:, :3])
                            x += velocity / n
                        images = vae.decode(x / LATENT_SCALE).sample.float().add(1).mul(0.5).clamp(0, 1)
                    for nfe, temperature, index, name in files:
                        if (nfe, temperature) == (n, temp):
                            to_pil_image(images[index].cpu()).save(stage / name)
            hashes = defaultdict(set)
            for n, temp, i, name in files:
                with Image.open(stage / name) as im:
                    if im.format != "PNG" or im.size != (256, 256):
                        raise ValueError("invalid sample output")
                    im.verify()
                if step == 0:
                    hashes[n, i].add(sha256((stage / name).read_bytes()).digest())
            if step == 0 and any(len(h) != 1 for h in hashes.values()):
                raise ValueError("step-zero outputs changed across addon temperatures")
            meta = {"step": step, "seed": 0, "cfg": 1.5, "labels": LABELS, "temperatures": TEMPERATURES,
                    "nfes": NFES, "sampler": "Euler", "files": [row[-1] for row in files]}
            (stage / "grid.json").write_text(json.dumps(meta))
            stage.rename(target)
    finally:
        model.train(was_training)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--run", type=Path, required=True)
    run = args.parse_args().run
    torch.manual_seed(0)
    model = VADiffusion().eval().to("cuda")
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(torch.count_nonzero(model.addon[k].project.weight) == 0 for k in LEVELS.values())
    # A zero-initialized addon must reproduce the exact frozen velocity at step 0.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.ones(1, 4, 32, 32, device="cuda")
        t = torch.tensor([0.35], device="cuda")
        y = torch.tensor([207], device="cuda")
        torch.testing.assert_close(model(x, t, y)[0], model.base(x, t, y), rtol=0, atol=0)
    generate_grid(model, load_vae("cuda"), run, 0)
    print(json.dumps({"step": 0, "images": 36, "base_frozen": True, "base_velocity_equal": True,
                      "addon_params": sum(p.numel() for p in model.addon.parameters())}), flush=True)
