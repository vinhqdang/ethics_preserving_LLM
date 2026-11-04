"""Check the location-scale / contamination theory against measured oracle-grouped robust-z firing.

For each framing f with harmful prevalence pi, the oracle rule standardises scores by the robust
(median, 1.4826 MAD) of the *mixture* (1-pi) F_b + pi F_h and fires above z*. Given the empirical
benign and harmful score distributions we predict its FPR and TPR in closed form and compare with the
non-transductive window measurements in results_v2/robust_*.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .lofo import MultiCells
from .recal import MAD_TO_SD, Config, fit_gate
from .robust_eval import SCENARIOS, build_pool


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(c, q)]


def mixture_stats(b, h, pi):
    x = np.r_[b, h]
    w = np.r_[np.full(len(b), (1 - pi) / len(b)), np.full(len(h), pi / len(h))]
    m = wquantile(x, w, 0.5)
    mad = wquantile(np.abs(x - m), w, 0.5)
    return m, MAD_TO_SD * mad


def predict(b, h, pi, zs):
    m, s = mixture_stats(b, h, pi)
    t = m + zs * s
    return float((b > t).mean()), float((h > t).mean()), m, s


def main(model, layer, tag):
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    tr, va = MultiCells(model, "train", SEEN_FAMILIES), MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.01)
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    pool = build_pool(model, fams, cfg, score)
    print(f"z*={zs:.3f}")
    out = {}
    for scen in ("uniform5", "realistic"):
        meas = json.load(open(ROOT / "results_v2" / f"robust_{tag}_{scen}{'_k30' if 'Phi' in tag else '_fresh'}.json"))["summary"]
        print(f"\n{scen}: family d' pi | pred FPR TPR | meas FPR TPR")
        for f in fams:
            b, h = pool.score[("benign", f)], pool.score[("harmful", f)]
            pi = SCENARIOS[scen](f)
            fpr, tpr, m, s = predict(b, h, pi, zs)
            dp = (h.mean() - b.mean()) / b.std()
            mf, mt = meas[f"family_z|benign|{f}"]["fire"][0], meas[f"family_z|harmful|{f}"]["fire"][0]
            print(f"{f:16s} {dp:5.2f} {pi:4.2f} | {fpr*100:5.1f} {tpr*100:5.1f} | {mf*100:5.1f} {mt*100:5.1f}")
            out[f"{scen}|{f}"] = dict(d=float(dp), pi=pi, pred=[fpr, tpr], meas=[mf, mt])
    json.dump(out, open(ROOT / "results_v2" / f"theory_check_{tag}.json", "w"), indent=1)
    a = np.array([v["pred"] + v["meas"] for v in out.values()])
    print("\nTPR  corr", np.corrcoef(a[:, 1], a[:, 3])[0, 1], "MAE", np.abs(a[:, 1] - a[:, 3]).mean())
    print("FPR  corr", np.corrcoef(a[:, 0], a[:, 2])[0, 1], "MAE", np.abs(a[:, 0] - a[:, 2]).mean())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[1].replace("/", "__"))
