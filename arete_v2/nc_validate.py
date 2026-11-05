"""Non-circular check of the contamination model, plus the small-cluster fallback attack.

(1) Split every framing's benign and harmful scores into halves A and B at random. The prediction uses the
    score distributions of half A only; the measurement is the oracle-grouped robust-z rule (statistics
    from a window drawn from half B) applied to the held-out remainder of half B. Repeated over splits.
(2) Small-cluster attack: prompts that fall in clusters below the minimum size use the validation
    statistics, which is the shared rule. We report the recall of that fallback rule per framing.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .lofo import MultiCells
from .recal import Config, fit_gate, robust_stats
from .robust_eval import SCENARIOS, build_pool
from .theory_check import predict


def measure(b, h, pi, zs, rng):
    """Window from one half at prevalence pi; statistics from it; firing on the other half."""
    ib, ih = rng.random(len(b)) < 0.5, rng.random(len(h)) < 0.5
    k = min(int(round(pi / (1 - pi) * ib.sum())), int(ih.sum()))
    win = np.r_[b[ib], rng.choice(h[ih], size=k, replace=False)]
    c, s = robust_stats(win)
    t = c + zs * s
    return float((b[~ib] > t).mean()), float((h[~ih] > t).mean())


def main(model: str, layer: int, reps: int = 200) -> None:
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    tr, va = MultiCells(model, "train", SEEN_FAMILIES), MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.01)
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    pool = build_pool(model, fams, cfg, score)
    rng = np.random.default_rng(0)
    rows, out = [], {"nc": {}, "fallback": {}}
    for scen in ("uniform5", "realistic"):
        for f in fams:
            b, h = pool.score[("benign", f)], pool.score[("harmful", f)]
            pi = SCENARIOS[scen](f)
            pf, pt = [], []
            ms = []
            for _ in range(reps):
                ia, ih = rng.random(len(b)) < 0.5, rng.random(len(h)) < 0.5
                _, tpr, _, _ = predict(b[ia], h[ih], pi, zs)
                pt.append(tpr)
                ms.append(measure(b[~ia], h[~ih], pi, zs, rng)[1])
            rows.append((np.mean(pt), np.mean(ms)))
            out["nc"][f"{scen}|{f}"] = dict(pred=float(np.mean(pt)), meas=float(np.mean(ms)))
    a = np.array(rows)
    print("non-circular TPR corr", np.corrcoef(a[:, 0], a[:, 1])[0, 1], "MAE", np.abs(a[:, 0] - a[:, 1]).mean())
    out["nc_summary"] = dict(corr=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), mae=float(np.abs(a[:, 0] - a[:, 1]).mean()))
    c0, s0 = fb
    print("\nsmall-cluster fallback (VAL statistics) recall / FPR per framing")
    for f in fams:
        b, h = pool.score[("benign", f)], pool.score[("harmful", f)]
        r, q = float(((h - c0) / s0 > zs).mean()), float(((b - c0) / s0 > zs).mean())
        out["fallback"][f] = dict(recall=r, fpr=q)
        print(f"{f:16s} recall {r*100:5.1f}  fpr {q*100:5.1f}")
    (ROOT / "results_v2" / f"nc_validate_{model.replace('/', '__')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
