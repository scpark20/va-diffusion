"""CNN VA-Diffusion addon for a frozen checkpointed SiT-S/2."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SiT


TOKEN_GRID = 16
TOKEN_DIM = 384
ENCODER_CONDITION_CHANNELS = 256
LATENT_DIM = 64


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class EncoderResBlock(nn.Module):
    """VDVAE-style bottleneck residual CNN block."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2
        self.net = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, 1),
            group_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            group_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            group_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def residual_stage(channels: int, blocks: int) -> nn.Sequential:
    return nn.Sequential(*(EncoderResBlock(channels) for _ in range(blocks)))


class VDVAEEncoder(nn.Module):
    """Bottom-up CNN encoder for clean SD-VAE latents [B,4,32,32]."""

    FEATURE_CHANNELS = {"e16": 512, "e8": 768, "e4": 1024}

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(4, 256, 3, padding=1)
        self.stage32 = residual_stage(256, 2)
        self.down16 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.stage16 = residual_stage(512, 4)
        self.down8 = nn.Conv2d(512, 768, 3, stride=2, padding=1)
        self.stage8 = residual_stage(768, 4)
        self.down4 = nn.Conv2d(768, 1024, 3, stride=2, padding=1)
        self.stage4 = residual_stage(1024, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 32, 32):
            raise ValueError(f"clean_latent must be [B,4,32,32], got {tuple(x.shape)}")
        x = self.stage32(self.stem(x))
        e16 = self.stage16(self.down16(x))
        e8 = self.stage8(self.down8(e16))
        e4 = self.stage4(self.down4(e8))
        return {"e4": e4, "e8": e8, "e16": e16}


class FeatureToCondition(nn.Module):
    """Convert a hierarchy feature to a 256-channel 16x16 condition map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, ENCODER_CONDITION_CHANNELS, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.proj(feature)
        if tuple(feature.shape[-2:]) != (TOKEN_GRID, TOKEN_GRID):
            feature = F.interpolate(
                feature,
                size=(TOKEN_GRID, TOKEN_GRID),
                mode="bilinear",
                align_corners=False,
            )
        return feature


class SpatialGaussianHead(nn.Module):
    """Residual CNN head producing spatial Gaussian mean and log standard deviation."""

    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        width = 768
        self.stem = nn.Conv2d(input_channels, width, 3, padding=1)
        self.blocks = residual_stage(width, 2)
        self.output_norm = group_norm(width)
        self.output = nn.Conv2d(width, 2 * latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.output(F.silu(self.output_norm(x)))


class VariationalAdapter(nn.Module):
    """CNN prior plus NVAE-style residual posterior at one SiT depth."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior = SpatialGaussianHead(TOKEN_DIM, latent_dim)
        self.posterior_delta = SpatialGaussianHead(
            TOKEN_DIM + ENCODER_CONDITION_CHANNELS,
            latent_dim,
        )
        self.project = nn.Conv2d(latent_dim, TOKEN_DIM, 1, bias=False)
        self._initialize_output_layers()

    def _initialize_output_layers(self) -> None:
        nn.init.zeros_(self.prior.output.weight)
        nn.init.zeros_(self.prior.output.bias)
        nn.init.zeros_(self.posterior_delta.output.weight)
        nn.init.normal_(
            self.posterior_delta.output.weight[: self.latent_dim],
            std=1e-3,
        )
        nn.init.zeros_(self.posterior_delta.output.bias)
        nn.init.zeros_(self.project.weight)

    @staticmethod
    def gaussian_kl(
        mu_q: torch.Tensor,
        logstd_q: torch.Tensor,
        mu_p: torch.Tensor,
        logstd_p: torch.Tensor,
    ) -> torch.Tensor:
        var_ratio = torch.exp(2.0 * (logstd_q - logstd_p))
        mean_term = (mu_q - mu_p).square() * torch.exp(-2.0 * logstd_p)
        return 0.5 * (var_ratio + mean_term - 1.0 + 2.0 * (logstd_p - logstd_q))

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_condition: Optional[torch.Tensor],
        prior_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tuple(hidden.shape[1:]) != (TOKEN_GRID * TOKEN_GRID, TOKEN_DIM):
            raise ValueError(f"SiT-S/2 hidden must be [B,256,384], got {tuple(hidden.shape)}")
        hidden_map = hidden.transpose(1, 2).reshape(
            hidden.shape[0], TOKEN_DIM, TOKEN_GRID, TOKEN_GRID
        )
        mu_p, logstd_p = self.prior(hidden_map).chunk(2, dim=1)
        logstd_p = logstd_p.clamp(-5.0, 2.0)

        if encoder_condition is not None:
            expected = (ENCODER_CONDITION_CHANNELS, TOKEN_GRID, TOKEN_GRID)
            if tuple(encoder_condition.shape[1:]) != expected:
                raise ValueError(
                    f"encoder condition must be [B,{expected[0]},16,16], "
                    f"got {tuple(encoder_condition.shape)}"
                )
            delta_mu, delta_logstd = self.posterior_delta(
                torch.cat([hidden_map, encoder_condition], dim=1)
            ).chunk(2, dim=1)
            mu_q = mu_p + delta_mu
            logstd_q = (logstd_p + delta_logstd).clamp(-5.0, 2.0)
            z = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
            kl = self.gaussian_kl(mu_q, logstd_q, mu_p, logstd_p).mean(dim=(1, 2, 3))
        else:
            if prior_noise is None:
                prior_noise = torch.zeros_like(mu_p)
            if prior_noise.shape != mu_p.shape:
                raise ValueError(f"prior_noise must be {tuple(mu_p.shape)}")
            z = mu_p + torch.exp(logstd_p) * prior_noise
            kl = hidden.new_zeros(hidden.shape[0])

        delta = self.project(z).flatten(2).transpose(1, 2)
        return hidden + delta, kl


class VADiffusionSiT(nn.Module):
    """Frozen checkpointed SiT-S/2 plus an approximately 100M-parameter CNN addon."""

    LEVEL_AT_BLOCK = {3: "e4", 7: "e8", 11: "e16"}

    def __init__(self, base: SiT, latent_dim: int = LATENT_DIM):
        super().__init__()
        if base.patch_size != 2 or len(base.blocks) != 12 or base.pos_embed.shape != (1, 256, 384):
            raise ValueError("base must be checkpointed SiT-S/2 at latent size 32")
        if latent_dim != LATENT_DIM:
            raise ValueError(f"this addon is fixed to latent_dim={LATENT_DIM}")

        self.base = base
        self.encoder = VDVAEEncoder()
        self.feature_to_condition = nn.ModuleDict({
            level: FeatureToCondition(channels)
            for level, channels in VDVAEEncoder.FEATURE_CHANNELS.items()
        })
        self.adapters = nn.ModuleDict({
            level: VariationalAdapter(latent_dim)
            for level in ("e4", "e8", "e16")
        })
        self.base.requires_grad_(False)

    def addon_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("base."):
                yield parameter

    def addon_parameter_counts(self) -> Dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "feature_condition": sum(p.numel() for p in self.feature_to_condition.parameters()),
            "prior": sum(p.numel() for adapter in self.adapters.values() for p in adapter.prior.parameters()),
            "posterior": sum(
                p.numel() for adapter in self.adapters.values() for p in adapter.posterior_delta.parameters()
            ),
            "project": sum(p.numel() for adapter in self.adapters.values() for p in adapter.project.parameters()),
        }

    def addon_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder.state_dict(),
            "feature_to_condition": self.feature_to_condition.state_dict(),
            "adapters": self.adapters.state_dict(),
        }

    def load_addon_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.feature_to_condition.load_state_dict(state["feature_to_condition"], strict=True)
        self.adapters.load_state_dict(state["adapters"], strict=True)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        clean_latent: Optional[torch.Tensor] = None,
        prior_noises: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.encoder(clean_latent) if clean_latent is not None else None
        prior_noises = {} if prior_noises is None else prior_noises

        hidden = self.base.x_embedder(x_t) + self.base.pos_embed
        condition = self.base.t_embedder(t) + self.base.y_embedder(labels, self.training)
        kl_by_level: Dict[str, torch.Tensor] = {}

        for block_index, block in enumerate(self.base.blocks):
            hidden = block(hidden, condition)
            level = self.LEVEL_AT_BLOCK.get(block_index)
            if level is None:
                continue
            encoder_condition = None
            if features is not None:
                encoder_condition = self.feature_to_condition[level](features[level])
            hidden, kl = self.adapters[level](
                hidden,
                encoder_condition,
                prior_noises.get(level),
            )
            kl_by_level[level] = kl

        prediction = self.base.final_layer(hidden, condition)
        prediction = self.base.unpatchify(prediction)
        if self.base.learn_sigma:
            prediction, _ = prediction.chunk(2, dim=1)
        return prediction, kl_by_level
