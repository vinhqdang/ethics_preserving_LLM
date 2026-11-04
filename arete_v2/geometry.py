"""ARETE v2 geometry: standardiser, intent/context hyperspheres, cap barrier, context subspace.

Differences from v1 (see review/audit_*.md):
  * caps are unsupervised prototypes (K configurable) - the real corpora carry no 19-way PKU labels;
  * the context sphere is supervised to encode the *framing family*, giving it a measurable role;
  * the context subspace Q is estimated from wrapper displacements h_wrapped - h_direct on TRAIN
    behaviours (not learned jointly), and steering is projected onto the true complement of Q;
  * steering follows a refusal direction (the only part of the intervention that can induce refusal).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def fit(h: torch.Tensor) -> "Standardizer":
        return Standardizer(h.mean(0), h.std(0).clamp_min(1e-4))

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return ((h - self.mean.to(h.device)) / self.std.to(h.device)).clamp(-8, 8)


def geodesic(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.acos((a * b).sum(-1).clamp(-1 + eps, 1 - eps))


class AreteGeometry(nn.Module):
    def __init__(self, dim: int, d_intent: int = 256, d_context: int = 256, n_caps: int = 8,
                 n_families: int = 4, hidden: int = 512):
        super().__init__()
        self.d_intent, self.d_context = d_intent, d_context
        self.f_i = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, d_intent))
        self.f_c = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, d_context))
        self.ln_i, self.ln_c = nn.LayerNorm(d_intent), nn.LayerNorm(d_context)
        self.family_head = nn.Linear(d_context, n_families)
        self.cap_w = nn.Parameter(F.normalize(torch.randn(n_caps, d_intent), dim=-1))
        self.cap_gamma = nn.Parameter(torch.full((n_caps,), 0.30))
        self.recon = nn.Linear(d_intent + d_context, dim, bias=False)

    def spheres(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zi = F.normalize(self.ln_i(self.f_i(x)), dim=-1)
        zc = F.normalize(self.ln_c(self.f_c(x)), dim=-1)
        return zi, zc

    def margins(self, zi: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.cap_w, dim=-1)
        return zi @ w.T - self.cap_gamma.clamp(0.05, 0.9)

    def max_margin(self, x: torch.Tensor) -> torch.Tensor:
        return self.margins(self.spheres(x)[0]).max(-1).values

    def retract(self, zi: torch.Tensor, slack: float = 0.05) -> torch.Tensor:
        """Nearest point of the safe set w.r.t. the most-violated cap (z-space only)."""
        m = self.margins(zi)
        k = m.argmax(-1)
        w = F.normalize(self.cap_w, dim=-1)[k]
        g = (self.cap_gamma.clamp(0.05, 0.9)[k] - slack).unsqueeze(-1)
        perp = F.normalize(zi - (zi * w).sum(-1, keepdim=True) * w, dim=-1, eps=1e-7)
        return F.normalize(g * w + (1 - g ** 2).clamp_min(1e-7).sqrt() * perp, dim=-1)

    def loss(self, x_clean, x_wrap, y, fam, *, margin=0.20, w_inv=1.0, w_cap=2.0, w_fam=0.5,
             w_orth=0.1, w_rec=0.05) -> dict[str, torch.Tensor]:
        zi_c, zc_c = self.spheres(x_clean)
        zi_w, zc_w = self.spheres(x_wrap)
        inv = geodesic(zi_c, zi_w).mean()
        harm = y > 0.5
        cap = x_clean.new_zeros(())
        for zi in (zi_c, zi_w):
            mm = self.margins(zi).max(-1).values
            cap = cap + 0.5 * (
                F.relu(margin - mm[harm]).mean() if harm.any() else 0.0
            ) + 0.5 * (F.relu(mm[~harm] + 0.10).mean() if (~harm).any() else 0.0)
        # context sphere must carry the framing family (0 = direct)
        fam_loss = F.cross_entropy(self.family_head(zc_w), fam) + F.cross_entropy(
            self.family_head(zc_c), torch.zeros_like(fam))
        zi_all, zc_all = torch.cat([zi_c, zi_w]), torch.cat([zc_c, zc_w])
        orth = ((zi_all.T @ zc_all) / zi_all.shape[0]).pow(2).sum()
        rec = 0.5 * (F.mse_loss(self.recon(torch.cat([zi_c, zc_c], -1)), x_clean)
                     + F.mse_loss(self.recon(torch.cat([zi_w, zc_w], -1)), x_wrap))
        total = w_inv * inv + w_cap * cap + w_fam * fam_loss + w_orth * orth + w_rec * rec
        return {"total": total, "inv": inv.detach(), "cap": cap.detach(), "fam": fam_loss.detach()}


def fit_geometry(x_clean, x_wrap, y, fam, *, n_families, epochs=40, bs=256, lr=1e-3, seed=0,
                 **kw) -> AreteGeometry:
    torch.manual_seed(seed)
    dev = "cuda"
    model = AreteGeometry(x_clean.shape[1], n_families=n_families, **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xc, xw, yy, ff = (t.to(dev) for t in (x_clean, x_wrap, y, fam))
    n = xc.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            out = model.loss(xc[idx], xw[idx], yy[idx], ff[idx])
            opt.zero_grad()
            out["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.eval()


def context_subspace(h_direct: torch.Tensor, h_wrapped: torch.Tensor, rank: int) -> torch.Tensor:
    """Orthonormal basis Q (D x rank) of the dominant wrapper displacement directions."""
    diff = (h_wrapped - h_direct)
    diff = diff - diff.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(diff, full_matrices=False)
    return vh[:rank].T.contiguous()


def refusal_direction(h_harm: torch.Tensor, h_benign: torch.Tensor) -> torch.Tensor:
    return F.normalize(h_harm.mean(0) - h_benign.mean(0), dim=0)


def project_out(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(I - Q Q^T) v with Q orthonormal - a genuine orthogonal projector."""
    return v - q @ (q.T @ v)
