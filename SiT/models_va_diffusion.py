"""SiT-S/2 instance of VA-Diffusion."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import SiT


class EncoderResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VDVAEEncoder(nn.Module):
    """Bottom-up encoder for clean SD-VAE latents [B, 4, 32, 32]."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(4, 64, 3, padding=1)
        self.stage32 = nn.Sequential(EncoderResBlock(64), EncoderResBlock(64))
        self.down16 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.stage16 = nn.Sequential(EncoderResBlock(128), EncoderResBlock(128))
        self.down8 = nn.Conv2d(128, 192, 3, stride=2, padding=1)
        self.stage8 = nn.Sequential(EncoderResBlock(192), EncoderResBlock(192))
        self.down4 = nn.Conv2d(192, 256, 3, stride=2, padding=1)
        self.stage4 = nn.Sequential(EncoderResBlock(256), EncoderResBlock(256))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 32, 32):
            raise ValueError(f"clean_latent must be [B,4,32,32], got {tuple(x.shape)}")
        x = self.stage32(self.stem(x))
        e16 = self.stage16(self.down16(x))
        e8 = self.stage8(self.down8(e16))
        e4 = self.stage4(self.down4(e8))
        return {"e4": e4, "e8": e8, "e16": e16}


class FeatureToTokens(nn.Module):
    """Project one encoder feature to the SiT-S/2 16x16 token grid."""

    def __init__(self, in_channels: int, token_channels: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, token_channels, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.proj(feature)
        if tuple(feature.shape[-2:]) != (16, 16):
            feature = F.interpolate(feature, size=(16, 16), mode="bilinear", align_corners=False)
        return feature.flatten(2).transpose(1, 2)


class TokenMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.hidden = nn.Linear(input_dim, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.hidden(self.norm(x))))


class VariationalAdapter(nn.Module):
    """Prior plus NVAE-style residual posterior."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior = TokenMLP(384, 2 * latent_dim)
        self.posterior_delta = TokenMLP(384 + 128, 2 * latent_dim)
        self.project = nn.Linear(latent_dim, 384, bias=False)
        self._initialize_output_layers()

    def _initialize_output_layers(self) -> None:
        nn.init.zeros_(self.prior.output.weight)
        nn.init.zeros_(self.prior.output.bias)

        nn.init.zeros_(self.posterior_delta.output.weight)
        nn.init.normal_(self.posterior_delta.output.weight[: self.latent_dim], std=1e-3)
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
        encoder_tokens: Optional[torch.Tensor],
        prior_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tuple(hidden.shape[1:]) != (256, 384):
            raise ValueError(f"SiT-S/2 hidden must be [B,256,384], got {tuple(hidden.shape)}")

        mu_p, logstd_p = self.prior(hidden).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-5.0, 2.0)

        if encoder_tokens is not None:
            if tuple(encoder_tokens.shape[1:]) != (256, 128):
                raise ValueError(
                    f"encoder tokens must be [B,256,128], got {tuple(encoder_tokens.shape)}"
                )
            delta_mu, delta_logstd = self.posterior_delta(
                torch.cat([hidden, encoder_tokens], dim=-1)
            ).chunk(2, dim=-1)
            mu_q = mu_p + delta_mu
            logstd_q = (logstd_p + delta_logstd).clamp(-5.0, 2.0)
            z = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
            kl = self.gaussian_kl(mu_q, logstd_q, mu_p, logstd_p).mean(dim=(1, 2))
        else:
            if prior_noise is None:
                prior_noise = torch.zeros_like(mu_p)
            if prior_noise.shape != mu_p.shape:
                raise ValueError(f"prior_noise must be {tuple(mu_p.shape)}")
            z = mu_p + torch.exp(logstd_p) * prior_noise
            kl = hidden.new_zeros(hidden.shape[0])

        return hidden + self.project(z), kl


class VADiffusionSiT(nn.Module):
    """Frozen checkpointed SiT-S/2 plus trainable VA-Diffusion addon."""

    LEVEL_AT_BLOCK = {3: "e4", 7: "e8", 11: "e16"}

    def __init__(self, base: SiT, latent_dim: int = 16):
        super().__init__()
        if base.patch_size != 2 or len(base.blocks) != 12 or base.pos_embed.shape != (1, 256, 384):
            raise ValueError("base must be checkpointed SiT-S/2 at latent size 32")

        self.base = base
        self.encoder = VDVAEEncoder()
        self.feature_to_tokens = nn.ModuleDict({
            "e4": FeatureToTokens(256),
            "e8": FeatureToTokens(192),
            "e16": FeatureToTokens(128),
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

    def addon_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "encoder": self.encoder.state_dict(),
            "feature_to_tokens": self.feature_to_tokens.state_dict(),
            "adapters": self.adapters.state_dict(),
        }

    def load_addon_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.encoder.load_state_dict(state["encoder"], strict=True)
        self.feature_to_tokens.load_state_dict(state["feature_to_tokens"], strict=True)
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
            encoder_tokens = None
            if features is not None:
                encoder_tokens = self.feature_to_tokens[level](features[level])
            hidden, kl = self.adapters[level](hidden, encoder_tokens, prior_noises.get(level))
            kl_by_level[level] = kl

        prediction = self.base.final_layer(hidden, condition)
        prediction = self.base.unpatchify(prediction)
        if self.base.learn_sigma:
            prediction, _ = prediction.chunk(2, dim=1)
        return prediction, kl_by_level
