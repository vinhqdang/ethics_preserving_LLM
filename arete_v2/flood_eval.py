"""Cluster purity and flooding analysis for the cluster-wise robust recalibration.

(1) Purity: fit the stream model on one realistic window and report, for every framing, the cluster that
    holds most of its prompts, that cluster's effective harmful share pi_c, and the adaptive-rule recall
    predicted from pi_c with the contamination formula (Proposition 3, empirical score distributions).
(2) Flooding: an attacker raises the harmful share of one target framing (its prompts stay in the same
    cluster) from 5% up to 50%. We measure recall of the shared, oracle-grouped, adaptive and hybrid
    rules on that framing, and compare the oracle-grouped rule with the contamination prediction.
Everything is detection level, non-transductive (statistics from half A, applied to half B).
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
from .robust_eval import DUAL_USE, SCENARIOS, build_pool, one_window
from .theory_check import predict

TARGETS = ("fiction", "code", "academic", "indirect", "past_tense")
LEVELS = (0.05, 0.1, 0.2, 0.3, 0.5)


def purity(pool, fams, scenario, rng, cfg, zs):
    n_h = len(next(v for (l, _), v in pool.score.items() if l == "harmful"))
    n_b = len(next(v for (l, _), v in pool.score.items() if l == "benign"))
    in_a_h, in_a_b = rng.random(n_h) < 0.5, rng.random(n_b) < 0.5
    s_w, f_w, fam_w, y_w = [], [], [], []
    for f in fams:
        b_idx = np.flatnonzero(in_a_b)
        p = scenario(f)
        k = int(round(p / (1 - p) * len(b_idx)))
        h_idx = rng.choice(np.flatnonzero(in_a_h), size=min(k, int(in_a_h.sum())), replace=False)
        for lab, idx, y in (("benign", b_idx, 0), ("harmful", h_idx, 1)):
            s_w.append(pool.score[(lab, f)][idx]); f_w.append(pool.feats[(lab, f)][idx])
            fam_w += [f] * len(idx); y_w += [y] * len(idx)
    s_w, f_w = np.concatenate(s_w), np.concatenate(f_w)
    fam_w, y_w = np.array(fam_w), np.array(y_w)
    sm = fit_stream_model(s_w, f_w, Config(score_layer=cfg.score_layer, score_view="mean",
                                           cluster_layer=cfg.cluster_layer))
    lab = sm.km.labels_
    rows = {}
    for f in fams:
        m = fam_w == f
        cnt = np.bincount(lab[m], minlength=lab.max() + 1)
        c = int(cnt.argmax())
        members = lab == c
        pi_c = float(y_w[members].mean())
        b, h = pool.score[("benign", f)], pool.score[("harmful", f)]
        fpr, tpr, _, _ = predict(b, h, pi_c, zs)
        rows[f] = dict(share=float(cnt[c] / m.sum()), n_cluster=int(members.sum()), pi_c=pi_c,
                       pi_nominal=scenario(f), pred_tpr=tpr, pred_fpr=fpr,
                       families_in_cluster=sorted(set(fam_w[members].tolist())))
    return rows


def main(model: str, layer: int, windows: int = 10) -> None:
    R.K_RANGE = range(2, 31)
    RG.K_RANGE = R.K_RANGE
    tag = model.replace("/", "__")
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    tr, va = MultiCells(model, "train", SEEN_FAMILIES), MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.01)
    _, thr_h, zs_h, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, 0.005)
    pool = build_pool(model, fams, cfg, score)
    print("fitted", flush=True)

    out = {"purity": {}, "flood": {}}
    for scen in ("realistic",):
        rows = purity(pool, fams, SCENARIOS[scen], np.random.default_rng(0), cfg, zs)
        out["purity"][scen] = rows
        print(f"\npurity ({scen}): family  share  n_cluster  pi_c  nominal | pred adaptive TPR")
        for f, r in rows.items():
            print(f"{f:18s} {r['share']:.2f} {r['n_cluster']:4d} {r['pi_c']:.2f} {r['pi_nominal']:.2f} | "
                  f"{r['pred_tpr']*100:5.1f}  {r['families_in_cluster']}", flush=True)

    for target in TARGETS:
        out["flood"][target] = {}
        b, h = pool.score[("benign", target)], pool.score[("harmful", target)]
        for p in LEVELS:
            scen = lambda f, t=target, pp=p: pp if f == t else (0.05 if f in DUAL_USE else 0.5)
            rng = np.random.default_rng(1)
            runs = [one_window(pool, fams, scen, rng, thr, thr_h, zs, zs_h, fb, cfg) for _ in range(windows)]
            rec = {r: float(np.mean([w[f"{r}|harmful|{target}"]["fire"] for w in runs]))
                   for r in ("shared", "family_z", "adaptive", "hybrid")}
            rec["pred_family_z"] = predict(b, h, p, zs)[1]
            out["flood"][target][str(p)] = rec
            print(f"flood {target:11s} pi={p:.2f}  " + "  ".join(f"{k}={v*100:5.1f}" for k, v in rec.items()),
                  flush=True)
    (ROOT / "results_v2" / f"flood_{tag}.json").write_text(json.dumps(out, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
