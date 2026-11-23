"""A frozen SiT-S/2 with three spatial, conditionally Gaussian latent groups."""
import os
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from models import SiT_S_2

WEIGHTS = Path(os.environ["VA_SIT_WEIGHTS"])
BASE_FILE = WEIGHTS / "transformer/diffusion_pytorch_model.safetensors"
LEVELS = ("e4", "e8", "e16")
AT_BLOCK = {4: "e4", 8: "e8", 12: "e16"}


class Residual(nn.Module):
    """VDVAE preactivation bottleneck, keeping the spatial resolution."""
    def __init__(self, width):
        super().__init__()
        mid = width // 2
        self.path = nn.Sequential(nn.GELU(), nn.Conv2d(width, mid, 1),
                                  nn.GELU(), nn.Conv2d(mid, mid, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(mid, mid, 3, padding=1),
                                  nn.GELU(), nn.Conv2d(mid, width, 1))

    def forward(self, x):
        return x + self.path(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleDict()
        in_channels = 4
        for res, width, depth in ((32, 256, 2), (16, 512, 4), (8, 768, 4), (4, 1024, 2)):
            layers = [] if res == 32 else [nn.AvgPool2d(2)]
            layers.append(nn.Conv2d(in_channels, width, 3, padding=1))
            layers.extend(Residual(width) for _ in range(depth))
            self.stages[f"e{res}"] = nn.Sequential(*layers)
            in_channels = width

    def forward(self, clean):
        features = {}
        for level, stage in self.stages.items():
            clean = stage(clean)
            if level in LEVELS:
                features[level] = clean
        return features


def gaussian_stats(channels):
    return nn.Sequential(nn.Conv2d(channels, 768, 3, padding=1), Residual(768),
                         Residual(768), nn.GELU(), nn.Conv2d(768, 128, 1))


class LatentGroup(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        self.feature = nn.Conv2d(encoder_channels, 256, 1)
        self.prior = gaussian_stats(384)
        self.correction = gaussian_stats(640)
        self.inject = nn.Conv2d(64, 384, 1, bias=False)
        nn.init.zeros_(self.inject.weight)
        for head in (self.prior, self.correction):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.normal_(self.correction[-1].weight[:64], std=0.001)

    def forward(self, tokens, encoding=None, noise=None):
        h = tokens.transpose(1, 2).reshape(tokens.shape[0], 384, 16, 16)
        mean, log_std = self.prior(h).float().chunk(2, dim=1)
        log_std = log_std.clamp(-5, 2)
        kl_sum = None
        if encoding is None:
            z = mean if noise is None else mean + log_std.exp() * noise
        else:
            e = F.interpolate(self.feature(encoding), size=(16, 16), mode="bilinear", align_corners=False)
            delta, scale_delta = self.correction(torch.cat((h, e), dim=1)).float().chunk(2, dim=1)
            q_log_std = (log_std + scale_delta).clamp(-5, 2)
            z = mean + delta + q_log_std.exp() * torch.randn_like(mean)
            twice_log_ratio = 2 * (q_log_std - log_std)
            kl = torch.expm1(twice_log_ratio) - twice_log_ratio + (delta * (-log_std).exp()).square()
            kl_sum = 0.5 * kl.sum(dim=(1, 2, 3))
        return tokens + self.inject(z).flatten(2).transpose(1, 2), kl_sum


class VASiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = SiT_S_2(input_size=32)
        state = load_file(str(BASE_FILE))
        self.base.load_state_dict(state, strict=True)
        if self.base.final_layer.linear.weight.count_nonzero() == 0:
            raise ValueError("base output head is untrained")
        self.base.requires_grad_(False).eval()
        self.addon = nn.ModuleDict({"encoder": Encoder(), "e4": LatentGroup(1024),
                                    "e8": LatentGroup(768), "e16": LatentGroup(512)})

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, xt, t, labels, clean=None, noises=None):
        encode = self.addon["encoder"](clean) if clean is not None else {}
        tokens = self.base.x_embedder(xt) + self.base.pos_embed
        condition = self.base.t_embedder(t) + self.base.y_embedder(labels, self.training)
        kl = {}
        for n, block in enumerate(self.base.blocks, 1):
            tokens = block(tokens, condition)
            if n in AT_BLOCK:
                level = AT_BLOCK[n]
                tokens, group_kl = self.addon[level](tokens, encode.get(level), (noises or {}).get(level))
                if group_kl is not None:
                    kl[level] = group_kl
        v = self.base.unpatchify(self.base.final_layer(tokens, condition))
        return v[:, :4], kl
