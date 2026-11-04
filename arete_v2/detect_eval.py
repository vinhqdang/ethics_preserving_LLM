"""Gate-level detection benchmark on held-out TEST behaviours (no generation).

Reports, per gate and framing family: harmful TPR, benign FPR, and AUROC of the underlying score,
over multiple seeds, with seen vs. unseen framing families separated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .attacks import FAMILIES, SEEN_FAMILIES, UNSEEN_FAMILIES
from .data import ROOT
from .defenses import Cells, build_bundle


def run(model: str, layer: int, seeds: list[int], target_fpr: float, rank: int, n_caps: int) -> dict:
    te = Cells(model, layer, "test")
    rows: dict[str, dict[str, list[float]]] = {}
    for seed in seeds:
        b = build_bundle(model, layer, seed=seed, rank=rank, n_caps=n_caps, target_fpr=target_fpr)
        for name, gate in b.gates.items():
            score = b.scores[name]
            for fam in FAMILIES:
                hh, bb = te.acts[("harmful", fam)], te.acts[("benign", fam)]
                with torch.no_grad():
                    g_h, g_b = gate(hh).float().mean().item(), gate(bb).float().mean().item()
                    s = torch.cat([score(hh).cpu(), score(bb).cpu()]).numpy()
                lab = np.r_[np.ones(len(hh)), np.zeros(len(bb))]
                auc = float(roc_auc_score(lab, s))
                r = rows.setdefault(f"{name}|{fam}", {"tpr": [], "fpr": [], "auc": []})
                r["tpr"].append(g_h); r["fpr"].append(g_b); r["auc"].append(auc)
    summary = {k: {m: [float(np.mean(v)), float(np.std(v))] for m, v in d.items()} for k, d in rows.items()}
    return summary


def print_table(summary: dict) -> None:
    gates = sorted({k.split("|")[0] for k in summary})
    print(f"{'gate':14s} {'family':20s} {'TPR':>14s} {'FPR(wrapped benign)':>22s} {'AUROC':>8s}")
    for g in gates:
        for fam in FAMILIES:
            r = summary[f"{g}|{fam}"]
            tag = "seen " if fam in SEEN_FAMILIES else "UNSEEN"
            print(f"{g:14s} {tag} {fam:16s} {r['tpr'][0]*100:6.1f}±{r['tpr'][1]*100:4.1f} "
                  f"{r['fpr'][0]*100:10.1f}±{r['fpr'][1]*100:4.1f} {r['auc'][0]:8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--caps", type=int, default=8)
    args = ap.parse_args()
    summary = run(args.model, args.layer, args.seeds, args.fpr, args.rank, args.caps)
    print_table(summary)
    out = ROOT / "results_v2"
    out.mkdir(exist_ok=True)
    tag = f"{args.model.replace('/', '__')}_L{args.layer}"
    (out / f"detect_{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
