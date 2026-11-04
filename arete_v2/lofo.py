"""Leave-one-framing-out (LOFO) model selection on VAL, using SEEN framings only.

For each seen wrapper w in {persona, refusal_suppression, prefix_injection}: train the gate on TRAIN
with the other seen framings, calibrate its threshold on VAL benign of those framings at a fixed FPR
budget, and evaluate TPR / FPR / AUROC on VAL prompts wrapped with the held-out w. Unseen framings and
the TEST split are never touched, so the configuration chosen here can be evaluated once on TEST as a
genuine out-of-distribution result.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import ROOT
from .defenses import calibrate, linear_probe_gate
from .extract_multi import POSITIONS, multi_dir
from .geometry import Standardizer

SEEN_WRAPPERS = ("persona", "refusal_suppression", "prefix_injection")


class MultiCells:
    """Multi-layer/position activations for one split: (label, family) -> [N, n_layers, 3, D]."""

    def __init__(self, model_name: str, split: str, families):
        d = multi_dir(model_name)
        self.acts = {}
        for label in ("harmful", "benign"):
            for fam in families:
                blob = torch.load(d / f"{split}_{label}_{fam}.pt")
                self.acts[(label, fam)] = blob["acts"]
                self.layers = blob["layers"]

    def get(self, label: str, fams, li: int, pi: int) -> torch.Tensor:
        return torch.cat([self.acts[(label, f)][:, li, pi].float() for f in fams])


def fit_probe(tr: MultiCells, fams, li: int, pi: int):
    hh, hb = tr.get("harmful", fams, li, pi), tr.get("benign", fams, li, pi)
    x = torch.cat([hh, hb])
    y = torch.cat([torch.ones(len(hh)), torch.zeros(len(hb))])
    std = Standardizer.fit(x)
    return linear_probe_gate(x, y, std)


def evaluate_split(score, cal_benign, eval_h, eval_b, fpr: float) -> dict:
    thr = calibrate(score(cal_benign).cpu(), fpr)
    s_h, s_b = score(eval_h).cpu(), score(eval_b).cpu()
    lab = np.r_[np.ones(len(s_h)), np.zeros(len(s_b))]
    return {"tpr": float((s_h > thr).float().mean()), "fpr": float((s_b > thr).float().mean()),
            "auc": float(roc_auc_score(lab, torch.cat([s_h, s_b]).numpy()))}


def lofo(model_name: str, fpr: float) -> dict:
    fams = ("direct",) + SEEN_WRAPPERS
    tr, va = MultiCells(model_name, "train", fams), MultiCells(model_name, "val", fams)
    out = {}
    for li, layer in enumerate(tr.layers):
        for pi, pos in enumerate(POSITIONS):
            folds = []
            for held in SEEN_WRAPPERS:
                seen = ["direct"] + [w for w in SEEN_WRAPPERS if w != held]
                score = fit_probe(tr, seen, li, pi)
                folds.append(evaluate_split(score, va.get("benign", seen, li, pi),
                                            va.get("harmful", [held], li, pi),
                                            va.get("benign", [held], li, pi), fpr))
            key = f"L{layer}|{pos}"
            out[key] = {m: float(np.mean([f[m] for f in folds])) for m in ("tpr", "fpr", "auc")}
            out[key]["folds"] = dict(zip(SEEN_WRAPPERS, folds))
            print(f"{key:12s} TPR={out[key]['tpr']:.3f} FPR={out[key]['fpr']:.3f} "
                  f"AUC={out[key]['auc']:.3f}  per-fold TPR=" +
                  " ".join(f"{f['tpr']:.2f}" for f in folds), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--fpr", type=float, default=0.01)
    args = ap.parse_args()
    res = lofo(args.model, args.fpr)
    dest = ROOT / "results_v2"
    dest.mkdir(exist_ok=True)
    (dest / f"lofo_{args.model.replace('/', '__')}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
