"""Learned soft-value critic for the value-gradient virtual teacher (MVP-2).

V_psi(x_t, t, c) regresses the terminal reward-to-go from a diffusion state. Because the
critic is a smooth learned function, grad_{x_t} V is a smooth, non-myopic improvement
direction -- unlike the raw (CLIP-adversarial, myopic) reward gradient that failed in MVP-1.
It is also reward-model-agnostic: it only consumes the SCALAR reward as a regression target,
so it works for non-differentiable rewards (OCR/GenEval) too.
"""
import math
import torch
import torch.nn as nn


def timestep_embedding(t, dim=256):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ValueNet(nn.Module):
    """Small spatially-structured conv critic. c = pooled text embedding (dim inferred lazily)."""

    def __init__(self, in_ch=16, hid=128, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, 2, 1), nn.SiLU(),  # 64 -> 32
            nn.Conv2d(hid, hid, 3, 2, 1), nn.SiLU(),    # 32 -> 16
            nn.Conv2d(hid, hid, 3, 2, 1), nn.SiLU(),    # 16 -> 8
        )
        self.cond = nn.Sequential(nn.LazyLinear(hid), nn.SiLU(), nn.Linear(hid, 2 * hid))
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1))

    def forward(self, x, t, c):
        h = self.enc(x.float())                      # [B, hid, 8, 8]
        h = h.mean(dim=(2, 3))                        # global pool -> [B, hid]
        temb = timestep_embedding(t, self.tdim)
        cond = self.cond(torch.cat([temb, c.float()], dim=-1))
        scale, shift = cond.chunk(2, dim=-1)
        h = h * (1 + scale) + shift                   # FiLM conditioning on (t, c)
        return self.head(h).squeeze(-1)               # [B]
