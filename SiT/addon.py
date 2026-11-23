"""Hierarchical CNN addon for the published SiT-S/2 ImageNet checkpoint."""
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from models import SiT_S_2

WEIGHTS = Path("/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256")
LEVELS = {3: "4", 7: "8", 11: "16"}


class Bottleneck(nn.Module):
    """VDVAE's four pre-activation convolutions and residual connection."""
    def __init__(self, width):
        super().__init__()
        middle = width // 2
        self.layers = nn.Sequential(
            nn.GELU(), nn.Conv2d(width, middle, 1),
            nn.GELU(), nn.Conv2d(middle, middle, 3, padding=1),
            nn.GELU(), nn.Conv2d(middle, middle, 3, padding=1),
            nn.GELU(), nn.Conv2d(middle, width, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleDict()
        incoming = 4
        for res, width, depth in [(32, 256, 2), (16, 512, 4), (8, 768, 4), (4, 1024, 2)]:
            self.stages[str(res)] = nn.Sequential(
                nn.Conv2d(incoming, width, 3, padding=1),
                *(Bottleneck(width) for _ in range(depth)))
            incoming = width

    def forward(self, x):
        features = {}
        for res, stage in self.stages.items():
            if res != "32":
                x = F.avg_pool2d(x, 2)
            x = stage(x)
            if res != "32":
                features[res] = x
        return features


def gaussian_head(incoming):
    return nn.Sequential(nn.Conv2d(incoming, 768, 3, padding=1), Bottleneck(768),
                         Bottleneck(768), nn.GELU(), nn.Conv2d(768, 128, 1))


class LatentGroup(nn.Module):
    def __init__(self, feature_width):
        super().__init__()
        self.condition = nn.Conv2d(feature_width, 256, 1)
        self.prior = gaussian_head(384)
        self.posterior_delta = gaussian_head(384 + 256)
        self.project = nn.Conv2d(64, 384, 1, bias=False)
        nn.init.zeros_(self.project.weight)
        for head in (self.prior, self.posterior_delta):
            nn.init.zeros_(head[-1].bias)
            nn.init.zeros_(head[-1].weight)
        nn.init.normal_(self.posterior_delta[-1].weight[:64], std=0.001)

    def forward(self, tokens, feature, prior_noise):
        h = tokens.transpose(1, 2).reshape(tokens.size(0), 384, 16, 16)
        mu, log_sd = self.prior(h).float().chunk(2, dim=1)
        log_sd = log_sd.clamp(-5, 2)
        kl = None
        if feature is None:
            z = mu if prior_noise is None else mu + log_sd.exp() * prior_noise
        else:
            e = F.interpolate(self.condition(feature), size=(16, 16), mode="bilinear", align_corners=False)
            dmu, dlog = self.posterior_delta(torch.cat((h, e), 1)).float().chunk(2, dim=1)
            qlog = (log_sd + dlog).clamp(-5, 2)
            z = mu + dmu + qlog.exp() * torch.randn_like(mu)
            ratio = 2 * (qlog - log_sd)
            kl = 0.5 * (torch.expm1(ratio) - ratio + (dmu * (-log_sd).exp()).square()).sum((1, 2, 3))
        return tokens + self.project(z).flatten(2).transpose(1, 2), kl


class VADiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = SiT_S_2(input_size=32)
        state = load_file(str(WEIGHTS / "transformer/diffusion_pytorch_model.safetensors"))
        self.base.load_state_dict(state, strict=True)
        self.base.requires_grad_(False).eval()
        self.addon = nn.ModuleDict({"encoder": ImageEncoder(), "4": LatentGroup(1024),
                                    "8": LatentGroup(768), "16": LatentGroup(512)})

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, t, labels, clean=None, noises=None):
        features = self.addon["encoder"](clean) if clean is not None else {}
        h = self.base.x_embedder(x) + self.base.pos_embed
        c = self.base.t_embedder(t) + self.base.y_embedder(labels, self.training)
        rates = {}
        for index, block in enumerate(self.base.blocks):
            h = block(h, c)
            if index in LEVELS:
                level = LEVELS[index]
                h, kl = self.addon[level](h, features.get(level), noises[level] if noises else None)
                if kl is not None:
                    rates[level] = kl
        output = self.base.unpatchify(self.base.final_layer(h, c))
        return output[:, :4], rates
