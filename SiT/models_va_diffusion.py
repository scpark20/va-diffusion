"""Frozen SiT-S/2 with three conditional spatial Gaussian adapters."""
import torch
from torch import nn
from torch.nn import functional as F

TOKEN_GRID, LATENT_DIM, WIDTH = 16, 64, 384
LEVELS = {3: "e4", 7: "e8", 11: "e16"}


class ResidualCNN(nn.Module):
    """VDVAE bottleneck: 1x1, 3x3, 3x3, 1x1."""
    def __init__(self, width):
        super().__init__()
        half = width // 2
        layers = []
        for cin, cout, k in [(width, half, 1), (half, half, 3),
                             (half, half, 3), (half, width, 1)]:
            layers.extend([nn.GroupNorm(32, cin), nn.SiLU(), nn.Conv2d(cin, cout, k, padding=k//2)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleDict()
        cin = 4
        for res, cout, depth in [(32, 256, 2), (16, 512, 4), (8, 768, 4), (4, 1024, 2)]:
            self.stages[f"e{res}"] = nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=1 if res == 32 else 2, padding=1),
                *(ResidualCNN(cout) for _ in range(depth)))
            cin = cout

    def forward(self, clean):
        features = {}
        for level, stage in self.stages.items():
            clean = stage(clean)
            if level != "e32":
                features[level] = clean
        return features


def gaussian_head(channels):
    return nn.Sequential(nn.Conv2d(channels, 768, 3, padding=1), ResidualCNN(768),
                         ResidualCNN(768), nn.GroupNorm(32, 768), nn.SiLU(),
                         nn.Conv2d(768, 2 * LATENT_DIM, 1))


class Adapter(nn.Module):
    def __init__(self, encoder_width):
        super().__init__()
        self.condition = nn.Conv2d(encoder_width, 256, 1)
        self.prior = gaussian_head(WIDTH)
        self.posterior = gaussian_head(WIDTH + 256)
        self.project = nn.Conv2d(LATENT_DIM, WIDTH, 1, bias=False)
        for head in (self.prior, self.posterior):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        nn.init.normal_(self.posterior[-1].weight[:LATENT_DIM], std=1e-3)
        nn.init.zeros_(self.project.weight)

    def forward(self, tokens, feature=None, noise=None):
        h = tokens.transpose(1, 2).reshape(-1, WIDTH, TOKEN_GRID, TOKEN_GRID)
        mu, logstd = self.prior(h).float().chunk(2, dim=1)
        logstd = logstd.clamp(-5, 2)
        if feature is None:
            if noise is not None and noise.shape != mu.shape:
                raise ValueError("prior noise shape must match the latent")
            z = mu if noise is None else mu + logstd.exp() * noise
            kl_sum = None
        else:
            e = F.interpolate(self.condition(feature), size=h.shape[-2:], mode="bilinear", align_corners=False)
            delta, delta_log = self.posterior(torch.cat((h, e), dim=1)).float().chunk(2, dim=1)
            q_log = (logstd + delta_log).clamp(-5, 2)
            z = mu + delta + q_log.exp() * torch.randn_like(mu)
            # Compute KL(q||p) in fp32 before summing all spatial/latent coordinates.
            d = 2 * (q_log - logstd)
            pointwise = 0.5 * (torch.expm1(d) - d + (delta * (-logstd).exp()).square())
            kl_sum = pointwise.sum(dim=(1, 2, 3))
        return tokens + self.project(z).flatten(2).transpose(1, 2), kl_sum


class VADiffusionSiT(nn.Module):
    def __init__(self, base):
        super().__init__()
        if (base.patch_size, len(base.blocks), tuple(base.pos_embed.shape)) != (2, 12, (1, 256, WIDTH)):
            raise ValueError("Expected SiT-S/2 with a 32x32 latent")
        self.base = base.eval().requires_grad_(False)
        self.addon = nn.ModuleDict({
            "encoder": Encoder(), "e4": Adapter(1024), "e8": Adapter(768), "e16": Adapter(512),
        })

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def load_addon_state_dict(self, state):
        self.addon.load_state_dict(state, strict=True)

    def forward(self, x_t, t, labels, clean_latent=None, prior_noises=None):
        features = {} if clean_latent is None else self.addon["encoder"](clean_latent)
        noises = {} if prior_noises is None else prior_noises
        h = self.base.x_embedder(x_t) + self.base.pos_embed
        c = self.base.t_embedder(t) + self.base.y_embedder(labels, self.training)
        sums = {}
        for i, block in enumerate(self.base.blocks):
            h = block(h, c)
            if i in LEVELS:
                level = LEVELS[i]
                h, kl_sum = self.addon[level](h, features.get(level), noises.get(level))
                if kl_sum is not None:
                    sums[level] = kl_sum
        velocity = self.base.unpatchify(self.base.final_layer(h, c))
        if self.base.learn_sigma:
            velocity = velocity.chunk(2, dim=1)[0]
        return velocity, sums
