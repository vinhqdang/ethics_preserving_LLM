"""Seed uncertainty of the deployed (k <= 30) gates.

Each seed s changes every random choice in the pipeline at once:
  * the probe is fit on a bootstrap resample of the training prompts (rng s), since the L2-regularised
    LBFGS fit itself is deterministic;
  * PCA / k-means / silhouette use random_state s, both for z* on VAL streams and in every window;
  * the A/B window splits and harmful subsamples use rng s.
For each seed, robust_eval windows give per-framing recall and FPR of the shared and hybrid rules;
we report the mean, standard deviation and range across seeds.
"""

from __future__ import annotations

import argparse
import functools
import json

import numpy as np
import torch

import arete_v2.recal as R
import arete_v2.recal_gen as RG
import arete_v2.robust_eval as RE
from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .defenses import linear_probe_gate
from .geometry import Standardizer
from .lofo import MultiCells
from .recal import Config, fit_gate

RULES = ("shared", "hybrid")


def bootstrap_probe(seed: int):
    """fit_probe replacement: same probe, trained on a bootstrap resample of each class."""
    rng = np.random.default_rng(seed)

    def fit(tr: MultiCells, fams, li: int, pi: int):
        hh, hb = tr.get("harmful", fams, li, pi), tr.get("benign", fams, li, pi)
        hh = hh[torch.from_numpy(rng.integers(0, len(hh), len(hh)))]
        hb = hb[torch.from_numpy(rng.integers(0, len(hb), len(hb)))]
        x = torch.cat([hh, hb])
        y = torch.cat([torch.ones(len(hh)), torch.zeros(len(hb))])
        return linear_probe_gate(x, y, Standardizer.fit(x))

    return fit


def run_seed(model: str, seed: int, windows: int, fpr: float, layer: int) -> dict:
    R.fit_probe = bootstrap_probe(seed)
    R.cluster = functools.partial(ORIG_CLUSTER, seed=seed)
    RE.fit_stream_model = functools.partial(ORIG_FSM, seed=seed)
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    tr = MultiCells(model, "train", SEEN_FAMILIES)
    va = MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr)
    _, thr_h, zs_h, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr / 2)
    pool = RE.build_pool(model, fams, cfg, score)
    out = {}
    for scen in ("uniform5", "realistic"):
        rng = np.random.default_rng(1000 + seed)
        runs = [RE.one_window(pool, fams, RE.SCENARIOS[scen], rng, thr, thr_h, zs, zs_h, fb, cfg)
                for _ in range(windows)]
        out[scen] = {f"{r}|{l}|{f}|{m}": float(np.mean([w[f"{r}|{l}|{f}"][m] for w in runs]))
                     for r in RULES for l in ("harmful", "benign") for f in fams
                     for m in ("fire", "asr") if m in runs[0][f"{r}|{l}|{f}"]}
    return out


ORIG_CLUSTER = R.cluster
ORIG_FSM = RG.fit_stream_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--kmax", type=int, default=30)
    args = ap.parse_args()
    R.K_RANGE = range(2, args.kmax + 1)
    RG.K_RANGE = R.K_RANGE
    tag = args.model.replace("/", "__")
    path = ROOT / "results_v2" / f"seed_check_{tag}.json"
    per_seed = json.loads(path.read_text())["per_seed"] if path.exists() else {}
    for s in range(args.seeds):
        if str(s) in per_seed:
            continue
        per_seed[str(s)] = run_seed(args.model, s, args.windows, args.fpr, args.layer)
        path.write_text(json.dumps({"per_seed": per_seed}, indent=1))
        print("seed", s, "done", flush=True)
    summary = {}
    for scen in ("uniform5", "realistic"):
        keys = per_seed["0"][scen]
        summary[scen] = {}
        for k in keys:
            v = np.array([per_seed[s][scen][k] for s in per_seed])
            summary[scen][k] = [float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())]
    path.write_text(json.dumps({"per_seed": per_seed, "summary": summary}, indent=1))
    fams = tuple(FAMILIES) + tuple(FRESH_FAMILIES)
    for scen in ("uniform5", "realistic"):
        print(f"\n[{scen}] mean +- sd (min-max) across {len(per_seed)} seeds, %")
        print(f"{'framing':20s} {'shared recall':>22s} {'hybrid recall':>22s} {'hybrid FPR':>20s} {'hybrid ASR':>18s}")
        for f in fams:
            S = summary[scen]

            def fmt(k, d=0):
                m, sd, lo, hi = (x * 100 for x in S[k])
                return f"{m:.{d}f}+-{sd:.{d + 1}f} ({lo:.{d}f}-{hi:.{d}f})"
            asr = fmt(f"hybrid|harmful|{f}|asr") if f"hybrid|harmful|{f}|asr" in S else "--"
            print(f"{f:20s} {fmt(f'shared|harmful|{f}|fire'):>22s} {fmt(f'hybrid|harmful|{f}|fire'):>22s} "
                  f"{fmt(f'hybrid|benign|{f}|fire', 1):>20s} {asr:>18s}", flush=True)


if __name__ == "__main__":
    main()
