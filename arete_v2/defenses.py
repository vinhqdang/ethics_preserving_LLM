"""Gated and always-on residual-stream defenses sharing one steering hook, plus calibration.

All gated defenses (ARETE variants, linear probe, MLP probe, CAST) are fit on the same TRAIN
activations, calibrated to the same benign false-positive budget on VAL, and steer with the same
refusal direction and hook, so differences come from the gate / projection only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import FAMILIES, SEEN_FAMILIES
from .extract import cache_dir
from .geometry import (AreteGeometry, Standardizer, context_subspace, fit_geometry,
                       project_out, refusal_direction)

GateFn = Callable[[torch.Tensor], torch.Tensor]  # raw last-token acts [B, D] -> bool [B]


class Cells:
    """Cached activations indexed by (label, family) for one split."""

    def __init__(self, model_name: str, layer: int, split: str):
        d = cache_dir(model_name, layer)
        self.acts: dict[tuple[str, str], torch.Tensor] = {}
        for label in ("harmful", "benign"):
            for fam in FAMILIES:
                blob = torch.load(d / f"{split}_{label}_{fam}.pt")
                self.acts[(label, fam)] = blob["acts"]

    def stack(self, label: str, families) -> torch.Tensor:
        return torch.cat([self.acts[(label, f)] for f in families])


@dataclass
class GatedSteer:
    layer: int
    gate_fn: GateFn
    direction: torch.Tensor  # raw-space unit vector (already null-space projected if applicable)
    alpha: float
    always_on: bool = False
    _gate: torch.Tensor | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._gate = None

    def edit(self, h: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill:
            if self.always_on:
                self._gate = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
            else:
                self._gate = self.gate_fn(h[:, -1, :].float()).to(h.device)
        if self._gate is None:
            return h
        delta = (self.alpha * self.direction).to(h.dtype).to(h.device)
        return h + self._gate.to(h.dtype)[:, None, None] * delta


# ---------------------------------------------------------------- gates
def arete_gate(geom: AreteGeometry, std: Standardizer, thr: float) -> GateFn:
    @torch.no_grad()
    def gate(h: torch.Tensor) -> torch.Tensor:
        return geom.max_margin(std(h.cuda())) > thr

    return gate


def linear_probe_gate(h_train, y_train, std: Standardizer, thr_logit: float = 0.0, steps=300):
    x = std(h_train).cuda()
    y = y_train.float().cuda()
    lin = nn.Linear(x.shape[1], 1).cuda()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(x).squeeze(-1), y) + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    def score(h):
        with torch.no_grad():
            return lin(std(h.cuda())).squeeze(-1)

    return score


def mlp_probe_gate(h_train, y_train, std: Standardizer, seed=0, epochs=60):
    torch.manual_seed(seed)
    x = std(h_train).cuda()
    y = y_train.float().cuda()
    net = nn.Sequential(nn.Linear(x.shape[1], 512), nn.SiLU(), nn.Linear(512, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device="cuda")
        for i in range(0, x.shape[0], 256):
            idx = perm[i : i + 256]
            loss = F.binary_cross_entropy_with_logits(net(x[idx]).squeeze(-1), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    def score(h):
        with torch.no_grad():
            return net(std(h.cuda())).squeeze(-1)

    return score


def cast_score(h_harm, h_benign) -> Callable[[torch.Tensor], torch.Tensor]:
    """CAST-style condition: cosine between (centred) activation and the harm-condition direction."""
    mu = torch.cat([h_harm, h_benign]).mean(0)
    cond = F.normalize(h_harm.mean(0) - h_benign.mean(0), dim=0).cuda()
    mu = mu.cuda()

    def score(h):
        return F.cosine_similarity(h.cuda() - mu, cond[None], dim=-1)

    return score


def calibrate(score_benign: torch.Tensor, target_fpr: float = 0.01) -> float:
    """Smallest threshold with benign FPR <= target on the calibration set."""
    k = int(torch.ceil(torch.tensor(len(score_benign) * (1 - target_fpr))).item()) - 1
    return float(torch.sort(score_benign.cpu()).values[min(max(k, 0), len(score_benign) - 1)]) + 1e-6


def to_gate(score_fn, thr: float) -> GateFn:
    return lambda h: score_fn(h) > thr


# ---------------------------------------------------------------- builders
@dataclass
class Bundle:
    """Everything fitted from TRAIN (+VAL calibration) for one (model, layer, seed)."""

    std: Standardizer
    geom: AreteGeometry
    q: torch.Tensor  # context subspace basis (D x r)
    refusal_dir: torch.Tensor  # raw space unit vector
    gates: dict[str, GateFn]
    scores: dict[str, Callable]


def build_bundle(model_name: str, layer: int, seed: int = 0, rank: int = 16, n_caps: int = 8,
                 target_fpr: float = 0.01) -> Bundle:
    tr = Cells(model_name, layer, "train")
    va = Cells(model_name, layer, "val")
    seen_wrapped = [f for f in SEEN_FAMILIES if f != "direct"]

    # standardiser from all train activations (seen families only: no OOD leakage)
    all_train = torch.cat([tr.stack("harmful", SEEN_FAMILIES), tr.stack("benign", SEEN_FAMILIES)])
    std = Standardizer.fit(all_train)

    # paired training tensors: (direct, wrapped) for every seen wrapper family
    xc, xw, ys, fs = [], [], [], []
    for fi, fam in enumerate(seen_wrapped, start=1):
        for label, y in (("harmful", 1.0), ("benign", 0.0)):
            xc.append(std(tr.acts[(label, "direct")]))
            xw.append(std(tr.acts[(label, fam)]))
            n = tr.acts[(label, "direct")].shape[0]
            ys.append(torch.full((n,), y))
            fs.append(torch.full((n,), fi, dtype=torch.long))
    geom = fit_geometry(torch.cat(xc), torch.cat(xw), torch.cat(ys), torch.cat(fs),
                        n_families=len(seen_wrapped) + 1, seed=seed, n_caps=n_caps)

    # context subspace from raw wrapper displacements (harmful + benign, seen wrappers)
    d_direct = torch.cat([tr.acts[(l, "direct")] for l in ("harmful", "benign") for _ in seen_wrapped])
    d_wrapped = torch.cat([tr.acts[(l, f)] for l in ("harmful", "benign") for f in seen_wrapped])
    q = context_subspace(d_direct, d_wrapped, rank)
    rdir = refusal_direction(tr.acts[("harmful", "direct")], tr.acts[("benign", "direct")])

    h_tr = torch.cat([tr.stack("harmful", SEEN_FAMILIES), tr.stack("benign", SEEN_FAMILIES)])
    y_tr = torch.cat([torch.ones(tr.stack("harmful", SEEN_FAMILIES).shape[0]),
                      torch.zeros(tr.stack("benign", SEEN_FAMILIES).shape[0])])
    scores: dict[str, Callable] = {
        "arete": lambda h, g=geom: g.max_margin(std(h.cuda())),
        "linear_probe": linear_probe_gate(h_tr, y_tr, std),
        "mlp_probe": mlp_probe_gate(h_tr, y_tr, std, seed=seed),
        "cast": cast_score(tr.stack("harmful", ["direct"]), tr.stack("benign", ["direct"])),
    }
    ben_val = va.stack("benign", SEEN_FAMILIES)  # calibrate on benign val (seen framings only)
    gates = {}
    for name, fn in scores.items():
        with torch.no_grad():
            thr = calibrate(fn(ben_val).cpu(), target_fpr)
        gates[name] = to_gate(fn, thr)
    return Bundle(std, geom, q, rdir, gates, scores)
