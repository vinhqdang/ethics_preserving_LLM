"""Saved artefacts for the VAL leave-one-framing-out numbers quoted in the paper.

  * projection: linear probe after projecting out a rank-r framing subspace (r in 0, 4, 16, 64),
    last-token layer-14 activations (the v1 cache);
  * multiview: union of per-position probes at layer 14, each calibrated at FPR/K;
  * hybrid: shared vs adaptive vs hybrid recalibration rules at prevalence 5%, 20%, 35%.
Writes results_v2/lofo_extra_<model>.json.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .defenses import Cells, calibrate, linear_probe_gate
from .extract_multi import POSITIONS
from .geometry import Standardizer, context_subspace
from .lofo import SEEN_WRAPPERS, MultiCells, fit_probe
from .recal import Config, fit_gate, run_stream

M = "Qwen/Qwen2.5-1.5B-Instruct"
FPR = 0.01


def _proj(h: torch.Tensor, q: torch.Tensor | None) -> torch.Tensor:
    return h if q is None else h - (h @ q) @ q.T


def projection() -> dict:
    tr, va = Cells(M, 14, "train"), Cells(M, 14, "val")
    out = {}
    for rank in (0, 4, 16, 64):
        folds = {}
        for held in SEEN_WRAPPERS:
            seen = ["direct"] + [w for w in SEEN_WRAPPERS if w != held]
            sw = [w for w in seen if w != "direct"]
            q = None
            if rank:
                dd = torch.cat([tr.acts[(l, "direct")] for l in ("harmful", "benign") for _ in sw])
                dw = torch.cat([tr.acts[(l, f)] for l in ("harmful", "benign") for f in sw])
                q = context_subspace(dd, dw, rank)
            hh, hb = tr.stack("harmful", seen), tr.stack("benign", seen)
            x = _proj(torch.cat([hh, hb]), q)
            y = torch.cat([torch.ones(len(hh)), torch.zeros(len(hb))])
            sc = linear_probe_gate(x, y, Standardizer.fit(x))
            thr = calibrate(sc(_proj(va.stack("benign", seen), q)).cpu(), FPR)
            s_h = sc(_proj(va.acts[("harmful", held)], q)).cpu()
            s_b = sc(_proj(va.acts[("benign", held)], q)).cpu()
            folds[held] = {"tpr": float((s_h > thr).float().mean()), "fpr": float((s_b > thr).float().mean()),
                           "auc": float(roc_auc_score(np.r_[np.ones(len(s_h)), np.zeros(len(s_b))],
                                                      torch.cat([s_h, s_b]).numpy()))}
        out[f"rank={rank}"] = {"folds": folds, **{m: float(np.mean([f[m] for f in folds.values()]))
                                                  for m in ("tpr", "fpr", "auc")}}
        print("projection", rank, {k: round(v, 3) for k, v in out[f"rank={rank}"].items() if k != "folds"})
    return out


def multiview() -> dict:
    fams = ("direct",) + SEEN_WRAPPERS
    tr, va = MultiCells(M, "train", fams), MultiCells(M, "val", fams)
    li = tr.layers.index(14)
    vidx = [POSITIONS.index(p) for p in POSITIONS]
    folds = {}
    for held in SEEN_WRAPPERS:
        seen = ["direct"] + [w for w in SEEN_WRAPPERS if w != held]
        fh = fb = None
        for pi in vidx:
            sc = fit_probe(tr, seen, li, pi)
            thr = calibrate(sc(va.get("benign", seen, li, pi)).cpu(), FPR / len(vidx))
            h = sc(va.get("harmful", [held], li, pi)).cpu() > thr
            b = sc(va.get("benign", [held], li, pi)).cpu() > thr
            fh, fb = (h, b) if fh is None else (fh | h, fb | b)
        folds[held] = {"tpr": float(fh.float().mean()), "fpr": float(fb.float().mean())}
    res = {"folds": folds, **{m: float(np.mean([f[m] for f in folds.values()])) for m in ("tpr", "fpr")}}
    print("multiview", {k: round(v, 3) for k, v in res.items() if k != "folds"})
    return res


def hybrid() -> dict:
    cfg = Config(score_view="mean", cluster_layer=14)
    fams = ("direct",) + SEEN_WRAPPERS
    tr, va = MultiCells(M, "train", fams), MultiCells(M, "val", fams)
    out = {}
    for pi in (0.05, 0.20, 0.35):
        res = {"shared": [], "adaptive": [], "hybrid": []}
        for held in SEEN_WRAPPERS:
            seen = ["direct"] + [w for w in SEEN_WRAPPERS if w != held]
            sc, thr, zs, fb = fit_gate(tr, va, seen, cfg, FPR)
            _, thr2, zs2, _ = fit_gate(tr, va, seen, cfg, FPR / 2)
            r = run_stream(sc, thr, zs, fb, va, fams, pi, 5, cfg, hybrid=(thr2, zs2))
            for k in res:
                res[k].append((r[f"{k}|{held}"]["tpr"], float(np.mean([r[f"{k}|{f}"]["fpr"] for f in fams]))))
        out[f"pi={pi}"] = {k: {"tpr": float(np.mean([a for a, _ in v])), "fpr": float(np.mean([b for _, b in v]))}
                           for k, v in res.items()}
        print("hybrid", pi, out[f"pi={pi}"])
    return out


def main() -> None:
    res = {"projection": projection(), "multiview": multiview(), "hybrid": hybrid()}
    (ROOT / "results_v2" / f"lofo_extra_{M.replace('/', '__')}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
