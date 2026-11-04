"""One-shot TEST evaluation of the pre-registered multi-view gate.

1. Detection: TPR / FPR / AUROC per framing family for the multi-view gate and each single view.
2. Generation: takes the undefended generations saved by gen_eval (``none|*``), regenerates only the
   prompts the multi-view gate fires on with the shared refusal-direction steer, and writes a new
   generations file (``none|*`` + ``multiview|*``) for judge.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .attacks import FAMILIES
from .data import ROOT, load_split
from .defenses import Cells, calibrate
from .extract_multi import extract
from .gen_eval import steer
from .geometry import refusal_direction
from .lofo import MultiCells, fit_probe
from .llm import generate, load_model
from .multiview import DEFAULT_LAYER, build_multiview

SINGLE_VIEWS = ("last", "tinst", "mean")


def detection(model_name: str, fpr: float) -> dict:
    from .attacks import SEEN_FAMILIES
    mv = build_multiview(model_name, target_fpr=fpr)
    tr = MultiCells(model_name, "train", SEEN_FAMILIES)
    va = MultiCells(model_name, "val", SEEN_FAMILIES)
    te = MultiCells(model_name, "test", tuple(FAMILIES))
    li = tr.layers.index(DEFAULT_LAYER)
    singles = {}
    for pi, v in enumerate(SINGLE_VIEWS):
        sc = fit_probe(tr, SEEN_FAMILIES, li, pi)
        with torch.no_grad():
            thr = calibrate(sc(va.get("benign", SEEN_FAMILIES, li, pi)).cpu(), fpr)
        singles[v] = (sc, thr, pi)
    out: dict = {}
    for fam in FAMILIES:
        ah, ab = te.acts[("harmful", fam)], te.acts[("benign", fam)]
        lab = np.r_[np.ones(len(ah)), np.zeros(len(ab))]
        with torch.no_grad():
            out[f"multiview|{fam}"] = {
                "tpr": float(mv.fire(ah).float().mean()), "fpr": float(mv.fire(ab).float().mean()),
                "auc": float(roc_auc_score(lab, torch.cat([mv.score(ah), mv.score(ab)]).numpy()))}
            for v, (sc, thr, pi) in singles.items():
                sh, sb = sc(ah[:, li, pi].float()).cpu(), sc(ab[:, li, pi].float()).cpu()
                out[f"probe_{v}|{fam}"] = {
                    "tpr": float((sh > thr).float().mean()), "fpr": float((sb > thr).float().mean()),
                    "auc": float(roc_auc_score(lab, torch.cat([sh, sb]).numpy()))}
    return out, mv


def generation(model_name: str, mv, gen_json: Path, gcg_json: Path | None, bs: int,
               tokens: int) -> dict:
    src = json.loads(gen_json.read_text(encoding="utf-8"))
    alpha = src["alpha"]
    harm, ben = load_split("test")
    tr = Cells(model_name, DEFAULT_LAYER, "train")
    rdir = refusal_direction(tr.acts[("harmful", "direct")], tr.acts[("benign", "direct")])
    te = MultiCells(model_name, "test", tuple(FAMILIES))
    model, tok = load_model(model_name)
    from .attacks import wrap
    gens = {k: v for k, v in src["generations"].items() if k.startswith("none|")}
    fire_rates = {}
    cells = {(lab, fam): [wrap(fam, p.text) for p in items]
             for lab, items in (("harmful", harm), ("benign", ben)) for fam in FAMILIES}
    acts = {k: te.acts[k] for k in cells}
    if gcg_json is not None and "gcg_requests" in src:
        blob = json.loads(gcg_json.read_text())
        texts = [blob[p.uid] for p in harm if p.uid in blob]
        cells[("harmful", "gcg")] = texts
        acts[("harmful", "gcg")] = extract(model, tok, texts, te.layers, 16)
    for (lab, fam), texts in cells.items():
        fired = mv.fire(acts[(lab, fam)]).numpy().astype(bool)
        fire_rates[f"{lab}|{fam}"] = float(fired.mean())
        outs = list(gens[f"none|{lab}|{fam}"])
        idx = np.flatnonzero(fired)
        if len(idx):
            sub = generate(model, tok, [texts[i] for i in idx], steer(DEFAULT_LAYER, rdir, alpha),
                           max_new_tokens=tokens, bs=bs)
            for i, s in zip(idx, sub):
                outs[i] = s
        gens[f"multiview|{lab}|{fam}"] = outs
        print(lab, fam, "fire", round(fire_rates[f"{lab}|{fam}"], 3), flush=True)
    return {"alpha": alpha, "requests": src["requests"], "gcg_requests": src.get("gcg_requests"),
            "generations": gens, "fire": fire_rates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--gen-json", required=True)
    ap.add_argument("--gcg", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()
    det, mv = detection(args.model, args.fpr)
    dest = ROOT / "results_v2"
    tag = args.model.replace("/", "__")
    (dest / f"mv_detect_{tag}.json").write_text(json.dumps(det, indent=1))
    for k, r in det.items():
        print(f"{k:32s} TPR={r['tpr']:.3f} FPR={r['fpr']:.3f} AUC={r['auc']:.3f}")
    res = generation(args.model, mv, Path(args.gen_json), Path(args.gcg) if args.gcg else None,
                     args.bs, args.tokens)
    (dest / f"gen_mv_{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
