"""Detection on TEST prompts re-wrapped with paraphrased (unseen) framing templates.

  --extract : cache TEST activations of every framing wrapped with para_templates.json
  --eval    : probe, shared threshold and z* fit exactly as before (original seen templates on
              TRAIN/VAL); windows and A/B splits as robust_eval, but the TEST pool is the paraphrased one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import arete_v2.recal as R
import arete_v2.recal_gen as RG
import arete_v2.robust_eval as RE
from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT, load_split
from .extract_multi import POSITIONS, extract, multi_dir
from .lofo import MultiCells
from .para_templates import PATH, wrap_para
from .recal import Config, fit_gate

FAMS = tuple(FAMILIES) + tuple(FRESH_FAMILIES)


def para_dir(model: str) -> Path:
    return Path(str(multi_dir(model)) + "_para")


def run_extract(model_name: str, layers: list[int], bs: int) -> None:
    from .llm import load_model
    table = json.loads(PATH.read_text(encoding="utf-8"))
    out = para_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model_name)
    harmful, benign = load_split("test")
    for label, items in (("harmful", harmful), ("benign", benign)):
        for fam in FAMS:
            path = out / f"test_{label}_{fam}.pt"
            if path.exists():
                continue
            acts = extract(model, tok, [wrap_para(fam, p.text, table) for p in items], layers, bs)
            torch.save({"acts": acts, "layers": layers, "positions": list(POSITIONS),
                        "uids": [p.uid for p in items]}, path)
            print("para", label, fam, tuple(acts.shape), flush=True)


def para_pool(model: str, cfg: Config, score_fn) -> RE.Pool:
    d = para_dir(model)
    score, feats = {}, {}
    for label in ("harmful", "benign"):
        for fam in FAMS:
            blob = torch.load(d / f"test_{label}_{fam}.pt")
            a, layers = blob["acts"], blob["layers"]
            sl, sp = layers.index(cfg.score_layer), POSITIONS.index(cfg.score_view)
            cl, cp = layers.index(cfg.cluster_layer), POSITIONS.index(cfg.cluster_view)
            with torch.no_grad():
                score[(label, fam)] = score_fn(a[:, sl, sp].float()).cpu().numpy()
            feats[(label, fam)] = a[:, cl, cp].float().numpy()
    return RE.Pool(score, feats, {}, {}, {}, {})


def auroc(pool: RE.Pool, fam: str) -> float:
    from sklearn.metrics import roc_auc_score
    h, b = pool.score[("harmful", fam)], pool.score[("benign", fam)]
    return float(roc_auc_score(np.r_[np.ones(len(h)), np.zeros(len(b))], np.r_[h, b]))


def run_eval(model: str, windows: int, fpr: float, layer: int) -> dict:
    cfg = Config(score_layer=layer, score_view="mean", cluster_layer=layer)
    tr, va = MultiCells(model, "train", SEEN_FAMILIES), MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr)
    _, thr_h, zs_h, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr / 2)
    res = {}
    for tag, pool in (("original", RE.build_pool(model, FAMS, cfg, score)),
                      ("para", para_pool(model, cfg, score))):
        res[tag] = {"auroc": {f: auroc(pool, f) for f in FAMS}}
        for scen in ("uniform5", "realistic"):
            rng = np.random.default_rng(0)
            runs = [RE.one_window(pool, FAMS, RE.SCENARIOS[scen], rng, thr, thr_h, zs, zs_h, fb, cfg)
                    for _ in range(windows)]
            res[tag][scen] = {k: float(np.mean([r[k]["fire"] for r in runs])) for k in runs[0]}
    for scen in ("uniform5", "realistic"):
        print(f"\n[{scen}] recall %, original -> paraphrased (benign FPR %)")
        print(f"{'framing':20s}{'AUROC':>14s}{'shared':>14s}{'oracle':>14s}{'hybrid':>14s}{'hyb FPR':>14s}")
        for f in FAMS:
            o, p = res["original"], res["para"]
            cell = lambda r, l="harmful", d=0: (f"{o[scen][f'{r}|{l}|{f}'] * 100:.{d}f}->"
                                               f"{p[scen][f'{r}|{l}|{f}'] * 100:.{d}f}")
            print(f"{f:20s}{o['auroc'][f]:7.2f}->{p['auroc'][f]:.2f}{cell('shared'):>14s}"
                  f"{cell('family_z'):>14s}{cell('hybrid'):>14s}{cell('hybrid', 'benign', 1):>14s}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.extract:
        run_extract(args.model, args.layers, args.bs)
    if args.eval:
        R.K_RANGE = range(2, args.kmax + 1)
        RG.K_RANGE = R.K_RANGE
        res = run_eval(args.model, args.windows, args.fpr, args.layer)
        tag = args.model.replace("/", "__")
        (ROOT / "results_v2" / f"para_eval_{tag}_k{args.kmax}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
