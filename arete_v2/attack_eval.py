"""Two extra checks on the cluster-wise recalibration (detection level, non-transductive A/B windows).

(1) Low-volume attack: the target framing contributes only n all-harmful prompts to the statistics window,
    so its clusters stay below the minimum size and fall back to the validation statistics.
(2) Baseline: per-cluster mean/std z-score instead of median/MAD. Its threshold is set so the benign
    FPR in the window equals the 1% budget, which uses labels and therefore favours the baseline.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import arete_v2.recal as R
import arete_v2.recal_gen as RG
from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .lofo import MultiCells
from .recal import Config, fit_gate
from .recal_gen import fit_stream_model
from .robust_eval import DUAL_USE, SCENARIOS, build_pool

TARGETS = ("fiction", "indirect", "academic", "code", "past_tense")
VOLUMES = (5, 15, 30, 60, 120)
MIN_CLUSTER = 30


def window(pool, fams, rng, in_a_b, in_a_h, scen, target=None, n_target=None):
    s, f, fam, y = [], [], [], []
    for fr in fams:
        b_idx = np.flatnonzero(in_a_b)
        if fr == target:
            b_idx = b_idx[:0]
            k = n_target
        else:
            p = scen(fr)
            k = int(round(p / (1 - p) * len(b_idx)))
        h_idx = rng.choice(np.flatnonzero(in_a_h), size=min(k, int(in_a_h.sum())), replace=False)
        for lab, idx, yy in (("benign", b_idx, 0), ("harmful", h_idx, 1)):
            s.append(pool.score[(lab, fr)][idx]); f.append(pool.feats[(lab, fr)][idx])
            fam += [fr] * len(idx); y += [yy] * len(idx)
    return np.concatenate(s), np.concatenate(f), np.array(fam), np.array(y)


def cluster_z(sm, s, feats, fb, zs, robust=True, mu_sd=None):
    c = sm.assign(feats)
    out = np.empty(len(s))
    for i, (si, ci) in enumerate(zip(s, c)):
        big = np.sum(sm.km.labels_ == ci) >= MIN_CLUSTER
        if robust:
            ctr, sc = sm.stats[ci] if big else fb
        else:
            ctr, sc = mu_sd[ci] if big else fb
        out[i] = (si - ctr) / max(sc, 1e-6)
    return out


def main(model: str, layer: int, windows: int = 10) -> None:
    R.K_RANGE = range(2, 31)
    RG.K_RANGE = R.K_RANGE
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    tr, va = MultiCells(model, "train", SEEN_FAMILIES), MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.01)
    _, thr_h, zs_h, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.005)
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    pool = build_pool(model, fams, cfg, score)
    scen = SCENARIOS["realistic"]
    n_h = len(next(v for (l, _), v in pool.score.items() if l == "harmful"))
    n_b = len(next(v for (l, _), v in pool.score.items() if l == "benign"))
    out = {"volume": {}, "baseline": {}}

    for t in TARGETS:
        out["volume"][t] = {}
        for n in VOLUMES:
            rng = np.random.default_rng(2)
            acc = []
            for _ in range(windows):
                ia_b, ia_h = rng.random(n_b) < 0.5, rng.random(n_h) < 0.5
                s_w, f_w, _, _ = window(pool, fams, rng, ia_b, ia_h, scen, t, n)
                sm = fit_stream_model(s_w, f_w, cfg)
                hb = pool.score[("harmful", t)][~ia_h]
                hf = pool.feats[("harmful", t)][~ia_h]
                cz = cluster_z(sm, hb, hf, fb, zs)
                acc.append(((cz > zs).mean(), ((hb > thr_h) | (cluster_z(sm, hb, hf, fb, zs_h) > zs_h)).mean()))
            a = np.mean(acc, axis=0)
            out["volume"][t][str(n)] = dict(adaptive=float(a[0]), hybrid=float(a[1]))
            print(f"volume {t:11s} n={n:4d} adaptive={a[0]*100:5.1f} hybrid={a[1]*100:5.1f}", flush=True)

    rng = np.random.default_rng(3)
    res = {}
    for _ in range(windows):
        ia_b, ia_h = rng.random(n_b) < 0.5, rng.random(n_h) < 0.5
        s_w, f_w, fam_w, y_w = window(pool, fams, rng, ia_b, ia_h, scen)
        sm = fit_stream_model(s_w, f_w, cfg)
        lab = sm.km.labels_
        mu_sd = {c: (s_w[lab == c].mean(), s_w[lab == c].std()) for c in np.unique(lab)}
        zw = cluster_z(sm, s_w, f_w, fb, zs, robust=False, mu_sd=mu_sd)
        zb = float(np.quantile(zw[y_w == 0], 0.99))
        for f in fams:
            for l, ia in (("harmful", ia_h), ("benign", ia_b)):
                s, ft = pool.score[(l, f)][~ia], pool.feats[(l, f)][~ia]
                r = (cluster_z(sm, s, ft, fb, zs) > zs).mean()
                m = (cluster_z(sm, s, ft, fb, zs, robust=False, mu_sd=mu_sd) > zb).mean()
                res.setdefault((l, f), []).append((r, m))
    print("\nbaseline: framing  robust(recall,fpr)  mean/std(recall,fpr)")
    for f in fams:
        h, b = np.mean(res[("harmful", f)], axis=0), np.mean(res[("benign", f)], axis=0)
        out["baseline"][f] = dict(robust=[float(h[0]), float(b[0])], meanstd=[float(h[1]), float(b[1])])
        print(f"{f:18s} {h[0]*100:5.1f} {b[0]*100:4.1f} | {h[1]*100:5.1f} {b[1]*100:4.1f}")
    (ROOT / "results_v2" / f"attack_{model.replace('/', '__')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
