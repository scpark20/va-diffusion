"""Exercise the production objective, checkpoint continuation and Euler sampler on GPU."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader

from sample_va_diffusion import generate_sample_set
from train_va_diffusion import (DATA, WEIGHTS, ImageNetStream, beta_at_step, make_model,
                               objective, restore_checkpoint, save_checkpoint, update)


def verify(device_index):
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(11)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    clean = torch.zeros(2, 4, 32, 32, device=device)
    t = torch.tensor([0., 1.], device=device)
    sums = {k: torch.full((2,), 4096., device=device) for k in ("e4", "e8", "e16")}
    loss, _ = objective(clean, clean, torch.ones_like(clean) * (1 - t[:, None, None, None]), t, sums, 5042)
    assert loss.item() == 3.5 and beta_at_step(21) == 1e-4 and beta_at_step(5041) < 1
    dataset = ImageNetStream(sorted(DATA.glob("train-*.parquet"))[:2], 11)
    loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(11))
    batches = iter(loader)
    batch1 = tuple(x.to(device) for x in next(batches))
    batch2 = tuple(x.to(device) for x in next(batches))
    del batches
    model = make_model(device).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        noise = torch.randn_like(clean)
        original = model.base(noise, t, batch1[1])
        initial, _ = model(noise, t, batch1[1], clean_latent=clean)
        torch.testing.assert_close(initial, original, rtol=0, atol=0)
    vae = AutoencoderKL.from_pretrained(str(WEIGHTS / "vae"), local_files_only=True).eval().requires_grad_(False).to(device)
    optimizer = torch.optim.AdamW(model.addon.parameters(), lr=1e-4, weight_decay=0)
    model.train()
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(p.requires_grad for p in model.addon.parameters())
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in model.addon.parameters()}
    frozen = {k: p.cpu().clone() for k, p in model.base.state_dict().items()}
    update(model, optimizer, vae, *batch1, 0)
    for k, p in model.base.state_dict().items():
        torch.testing.assert_close(p.cpu(), frozen[k], rtol=0, atol=0)
    assert all(p.grad is None for p in model.base.parameters())
    for k in ("e4", "e8", "e16"):
        assert torch.count_nonzero(model.addon[k].project.weight)
    position = {"epoch": 0, "batch_in_epoch": 1, "global_step": 1}
    with TemporaryDirectory(prefix="va-diffusion-check-") as directory:
        tmp = Path(directory)
        path = tmp / "latest.pt"
        save_checkpoint(path, model, position, {})
        on_disk = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        assert set(on_disk) == {"format", "addon", "settings", "epoch", "batch_in_epoch", "global_step", "cpu_rng", "gpu_rng"}
        assert set(on_disk["addon"]) == set(model.addon.state_dict())
        assert path.stat().st_size < sum(v.nbytes for v in on_disk["addon"].values()) + 1024**2
        del on_disk
        # The reference follows the same weights-only restart with fresh AdamW moments.
        optimizer = torch.optim.AdamW(model.addon.parameters(), lr=1e-4, weight_decay=0)
        expected = update(model, optimizer, vae, *batch2, 1)
        reference = {k: v.detach().cpu() for k, v in model.addon.state_dict().items()}
        del model, optimizer
        model = make_model(device).train()
        optimizer = torch.optim.AdamW(model.addon.parameters(), lr=1e-4, weight_decay=0)
        assert restore_checkpoint(path, model, {}) == position
        repeated = iter(loader)
        next(repeated)
        for seen, known in zip(next(repeated), batch2):
            torch.testing.assert_close(seen, known.cpu(), rtol=0, atol=0)
        del repeated
        actual = update(model, optimizer, vae, *batch2, 1)
        torch.testing.assert_close(torch.tensor(actual["loss"]), torch.tensor(expected["loss"]), rtol=0, atol=0)
        for k, v in model.addon.state_dict().items():
            torch.testing.assert_close(v.cpu(), reference[k], rtol=0, atol=0, msg=k)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        _, names = generate_sample_set(model, vae, tmp / "samples", 2)
        assert model.training and torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        assert len(set(names)) == 36
        for name in names:
            with Image.open(tmp / "samples" / name) as im:
                assert im.format == "PNG" and im.size == (256, 256)
                im.verify()
        print(json.dumps({"base_output_equal": True, "base_weights_unchanged": True,
                          "fresh_adam_resume_equal": True, "verified_samples": len(names),
                          "addon_parameters": sum(p.numel() for p in model.addon.parameters()),
                          "checkpoint_bytes": path.stat().st_size, "data_mse": actual["data_mse"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    verify(parser.parse_args().device)
