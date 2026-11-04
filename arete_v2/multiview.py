"""Multi-view gate: union of per-position linear probes, each calibrated at FPR/K.

Selected by leave-one-framing-out on VAL (see lofo.py) *before* touching TEST:
layer 14, views = (last, tinst, mean). Different token positions fail on different framings
(``last`` misses prefix-injection, ``tinst``/``mean`` miss persona), so a union of independently
calibrated views keeps the benign false-positive budget (Bonferroni) while covering more framings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .attacks import SEEN_FAMILIES
from .defenses import calibrate
from .extract_multi import POSITIONS
from .lofo import MultiCells, fit_probe

DEFAULT_LAYER = 14
DEFAULT_VIEWS = ("last", "tinst", "mean")


@dataclass(frozen=True)
class MultiViewGate:
    layer_index: int
    view_index: tuple[int, ...]
    scores: tuple[Callable[[torch.Tensor], torch.Tensor], ...]
    thresholds: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]

    def fire(self, acts: torch.Tensor) -> torch.Tensor:
        """acts: [N, n_layers, n_positions, D] -> bool [N]."""
        hits = [s(acts[:, self.layer_index, p].float()).cpu() > t
                for s, p, t in zip(self.scores, self.view_index, self.thresholds)]
        return torch.stack(hits).any(0)

    def score(self, acts: torch.Tensor) -> torch.Tensor:
        """Continuous score for AUROC: max over views of benign-standardised probe logits."""
        z = [(s(acts[:, self.layer_index, p].float()).cpu() - c) / sc
             for s, p, c, sc in zip(self.scores, self.view_index, self.centers, self.scales)]
        return torch.stack(z).max(0).values


def build_multiview(model_name: str, layer: int = DEFAULT_LAYER, views=DEFAULT_VIEWS,
                    target_fpr: float = 0.01) -> MultiViewGate:
    tr = MultiCells(model_name, "train", SEEN_FAMILIES)
    va = MultiCells(model_name, "val", SEEN_FAMILIES)
    li = tr.layers.index(layer)
    vidx = tuple(POSITIONS.index(v) for v in views)
    scores, thrs, centers, scales = [], [], [], []
    for pi in vidx:
        sc = fit_probe(tr, SEEN_FAMILIES, li, pi)
        with torch.no_grad():
            cal = sc(va.get("benign", SEEN_FAMILIES, li, pi)).cpu()
        scores.append(sc)
        thrs.append(calibrate(cal, target_fpr / len(vidx)))
        centers.append(float(cal.mean()))
        scales.append(float(cal.std()))
    return MultiViewGate(li, vidx, tuple(scores), tuple(thrs), tuple(centers), tuple(scales))
