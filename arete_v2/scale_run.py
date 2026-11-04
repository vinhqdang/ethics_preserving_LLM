"""End-to-end replication on a new target model with the configuration frozen from the 1.5B study.

Frozen (pre-registered) configuration, chosen on Qwen2.5-1.5B and applied unchanged here:
  user-mean logistic probe at the middle layer, clustering on the same features, PCA-32,
  k <= 30 by silhouette (k <= 12 also reported), min cluster 30, 1% FPR budget, hybrid floor,
  steer alpha = 0.4 x mean activation norm along the plain-request refusal direction.
Every step writes its output before the next starts, so a preempted run can resume.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig

from .attacks import ALL_FAMILIES, SEEN_FAMILIES, wrap_any
from .data import ROOT, load_split
from .defenses import calibrate
from .extract_multi import POSITIONS, extract, multi_dir
from .gen_eval import steer
from .llm import generate, load_model
from .lofo import MultiCells, fit_probe

RES = ROOT / "results_v2"


def layers_for(model_name: str) -> tuple[list[int], int]:
    n = AutoConfig.from_pretrained(model_name).num_hidden_layers
    return [n // 4, n // 2, (3 * n) // 4], n // 2


def step_extract(model, tok, model_name: str, layers: list[int], bs: int) -> None:
    out = multi_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        harmful, benign = load_split(split)
        fams = ALL_FAMILIES if split == "test" else SEEN_FAMILIES
        for label, items in (("harmful", harmful), ("benign", benign)):
            for fam in fams:
                path = out / f"{split}_{label}_{fam}.pt"
                if path.exists():
                    continue
                acts = extract(model, tok, [wrap_any(fam, p.text) for p in items], layers, bs)
                torch.save({"acts": acts, "layers": layers, "positions": list(POSITIONS),
                            "uids": [p.uid for p in items]}, path)
                print("extract", split, label, fam, tuple(acts.shape), flush=True)


def steer_params(model_name: str, mid: int) -> tuple[torch.Tensor, float]:
    tr = MultiCells(model_name, "train", SEEN_FAMILIES)
    li, pi = tr.layers.index(mid), POSITIONS.index("last")
    h = tr.acts[("harmful", "direct")][:, li, pi].float()
    b = tr.acts[("benign", "direct")][:, li, pi].float()
    rdir = torch.nn.functional.normalize(h.mean(0) - b.mean(0), dim=0)
    norm = tr.get("harmful", SEEN_FAMILIES, li, pi).norm(dim=-1).mean()
    return rdir, float(0.4 * norm)


def step_generate(model, tok, model_name: str, mid: int, bs: int, tokens: int) -> Path:
    tag = model_name.replace("/", "__")
    path = RES / f"gen_all_{tag}.json"
    harm, ben = load_split("test")
    blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "requests": {"harmful": [p.text for p in harm], "benign": [p.text for p in ben]}, "generations": {}}
    rdir, alpha = steer_params(model_name, mid)
    blob["alpha"] = alpha
    gens = blob["generations"]
    for lab, items in (("harmful", harm), ("benign", ben)):
        for fam in ALL_FAMILIES:
            if f"always_on|{lab}|{fam}" in gens:
                continue
            texts = [wrap_any(fam, p.text) for p in items]
            gens[f"none|{lab}|{fam}"] = generate(model, tok, texts, None, tokens, bs)
            gens[f"always_on|{lab}|{fam}"] = generate(model, tok, texts, steer(mid, rdir.cuda(), alpha),
                                                      tokens, bs)
            path.write_text(json.dumps(blob), encoding="utf-8")
            print("generate", lab, fam, flush=True)
    return path


def step_diagnose(model_name: str, mid: int) -> None:
    tr = MultiCells(model_name, "train", SEEN_FAMILIES)
    va = MultiCells(model_name, "val", SEEN_FAMILIES)
    te = MultiCells(model_name, "test", tuple(ALL_FAMILIES))
    li = tr.layers.index(mid)
    out = {}
    for view in ("last", "mean"):
        pi = POSITIONS.index(view)
        sc = fit_probe(tr, SEEN_FAMILIES, li, pi)
        with torch.no_grad():
            thr = calibrate(sc(va.get("benign", SEEN_FAMILIES, li, pi)).cpu(), 0.01)
            for fam in ALL_FAMILIES:
                sh = sc(te.acts[("harmful", fam)][:, li, pi].float()).cpu()
                sb = sc(te.acts[("benign", fam)][:, li, pi].float()).cpu()
                out[f"{view}|{fam}"] = {
                    "auc": float(roc_auc_score(np.r_[np.ones(len(sh)), np.zeros(len(sb))],
                                               torch.cat([sh, sb]).numpy())),
                    "shared_tpr": float((sh > thr).float().mean()),
                    "shared_fpr": float((sb > thr).float().mean()),
                    "oracle_tpr": float((sh > calibrate(sb, 0.01)).float().mean())}
    (RES / f"diag_{model_name.replace('/', '__')}.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print("diag", k, {m: round(x, 3) for m, x in v.items()}, flush=True)


def step_robust(model_name: str, mid: int, windows: int) -> None:
    import arete_v2.recal as R
    import arete_v2.recal_gen as RG
    from .robust_eval import evaluate
    tag = model_name.replace("/", "__")
    for kmax in (30, 12):
        R.K_RANGE = range(2, kmax + 1)
        RG.K_RANGE = R.K_RANGE
        for scen in ("realistic", "uniform5"):
            res = evaluate(model_name, scen, windows, 0.01, True, layer=mid)
            res["k_max"] = kmax
            (RES / f"robust_{tag}_{scen}_k{kmax}.json").write_text(json.dumps(res, indent=1))
            print("robust done", scen, kmax, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--steps", nargs="+", default=["extract", "generate", "judge", "diagnose", "robust"])
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    layers, mid = layers_for(args.model)
    print("layers", layers, "mid", mid, flush=True)
    if {"extract", "generate"} & set(args.steps):
        model, tok = load_model(args.model, torch.bfloat16)
        if "extract" in args.steps:
            step_extract(model, tok, args.model, layers, 32)
        if "generate" in args.steps:
            step_generate(model, tok, args.model, mid, args.bs, args.tokens)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    tag = args.model.replace("/", "__")
    if "judge" in args.steps:
        from .judge import judge_file
        src = RES / f"gen_all_{tag}.json"
        judged = judge_file(src, 16)
        (RES / f"gen_all_{tag}_judged.json").write_text(json.dumps(judged), encoding="utf-8")
        print("judge done", flush=True)
    if "diagnose" in args.steps:
        step_diagnose(args.model, mid)
    if "robust" in args.steps:
        step_robust(args.model, mid, args.windows)
    print("ALL_DONE", args.model, flush=True)


if __name__ == "__main__":
    main()
