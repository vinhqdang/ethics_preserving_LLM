"""Defended-generation benchmark on held-out TEST behaviours.

For each framing family: attack success rate (ASR = harmful prompt NOT refused) and
over-refusal rate (benign prompt refused). Gated defenses only change the prompts their gate
fires on, so unfired prompts reuse the undefended generation (exactly equivalent, much cheaper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .attacks import FAMILIES, SEEN_FAMILIES, wrap
from .data import ROOT, load_split
from .defenses import Cells, GatedSteer, build_bundle
from .extract import cache_dir
from .geometry import project_out
from .llm import generate, is_refusal, load_model

GATED = ("arete", "linear_probe", "mlp_probe", "cast")


def mean_norm(model_name: str, layer: int) -> float:
    tr = Cells(model_name, layer, "train")
    return float(tr.stack("harmful", SEEN_FAMILIES).norm(dim=-1).mean())


def steer(layer, direction, alpha, always_on=True) -> GatedSteer:
    return GatedSteer(layer=layer, gate_fn=lambda h: torch.ones(h.shape[0], dtype=torch.bool),
                      direction=direction.cuda(), alpha=alpha, always_on=always_on)


def run(args) -> dict:
    harm, ben = load_split("test")
    harm, ben = harm[: args.limit], ben[: args.limit]
    te = Cells(args.model, args.layer, "test")
    model, tok = load_model(args.model, torch.bfloat16 if args.bf16 else torch.float16)
    alpha = args.alpha_rel * mean_norm(args.model, args.layer)
    gen = lambda texts, d=None: generate(model, tok, texts, d, max_new_tokens=args.tokens, bs=args.bs)

    # cells: (label, family) -> list of prompt texts
    cells = {(lab, fam): [wrap(fam, p.text) for p in items]
             for lab, items in (("harmful", harm), ("benign", ben)) for fam in FAMILIES}
    extra = {}
    if args.gcg:
        blob = json.loads(Path(args.gcg).read_text())
        by_uid = {u: t for u, t in blob.items()}
        uid_h = [p.uid for p in harm if p.uid in by_uid]
        cells[("harmful", "gcg")] = [by_uid[u] for u in uid_h]
        extra["gcg_requests"] = [p.text for p in harm if p.uid in by_uid]
        extra["gcg_n"] = len(uid_h)
    print("cells", {k: len(v) for k, v in cells.items()}, "alpha", alpha, flush=True)

    base = {k: gen(v) for k, v in cells.items()}
    print("baseline done", flush=True)

    def rates(outs_by_cell) -> dict:
        r = {}
        for (lab, fam), outs in outs_by_cell.items():
            ref = np.array([is_refusal(o) for o in outs], dtype=float)
            r[f"{lab}|{fam}"] = float(1 - ref.mean()) if lab == "harmful" else float(ref.mean())
        return r  # harmful -> ASR, benign -> over-refusal

    result: dict = {"alpha": alpha, "none": rates(base), **extra}
    result["requests"] = {"harmful": [p.text for p in harm], "benign": [p.text for p in ben]}
    gens: dict[str, list[str]] = {f"none|{lab}|{fam}": o for (lab, fam), o in base.items()}
    result["generations"] = gens
    rdir_cache = {}
    always = {}
    for seed in args.seeds:
        b = build_bundle(args.model, args.layer, seed=seed, rank=args.rank, n_caps=args.caps,
                         target_fpr=args.fpr)
        rdir_cache[seed] = b
        if seed == args.seeds[0]:
            always = {k: gen(v, steer(args.layer, b.refusal_dir, alpha)) for k, v in cells.items()}
            result["always_on"] = rates(always)
            gens.update({f"always_on|{lab}|{fam}": o for (lab, fam), o in always.items()})
            print("always_on done", flush=True)
        for name in GATED:
            direction = project_out(b.refusal_dir, b.q) if name == "arete" else b.refusal_dir
            if name == "arete":
                direction = direction / direction.norm()
            outs = {}
            for (lab, fam), texts in cells.items():
                if fam == "gcg":
                    acts = extract_gcg(model, tok, texts, args.layer)
                else:
                    acts = te.acts[(lab, fam)][: len(texts)]
                with torch.no_grad():
                    fired = b.gates[name](acts).cpu().numpy().astype(bool)
                o = list(base[(lab, fam)])
                idx = np.flatnonzero(fired)
                if len(idx):
                    sub = gen([texts[i] for i in idx], steer(args.layer, direction, alpha))
                    for i, s in zip(idx, sub):
                        o[i] = s
                outs[(lab, fam)] = o
                result.setdefault(f"fire|{name}|{lab}|{fam}", []).append(float(fired.mean()))
            result.setdefault(name, []).append(rates(outs))
            gens.update({f"{name}|s{seed}|{lab}|{fam}": o for (lab, fam), o in outs.items()})
            print("seed", seed, name, "done", flush=True)
    return result


def extract_gcg(model, tok, texts, layer):
    from .llm import extract_acts
    return extract_acts(model, tok, texts, layer, 16)


def summarize(result: dict) -> None:
    def fmt(vals):
        a = np.array(vals); return f"{a.mean()*100:5.1f}±{a.std()*100:3.1f}"
    keys = [k for k in result["none"]]
    print(f"{'defense':14s}" + "".join(f"{k.split('|')[1][:9]:>11s}" for k in keys if k.startswith("harmful")))
    for name in ("none", "always_on") + GATED:
        row = result[name]
        rows = row if isinstance(row, list) else [row]
        h = [k for k in keys if k.startswith("harmful")]
        print(f"{name:14s}" + "".join(f"{fmt([r[k] for r in rows]):>11s}" for k in h), " (ASR %)")
    for name in ("none", "always_on") + GATED:
        row = result[name]
        rows = row if isinstance(row, list) else [row]
        b = [k for k in keys if k.startswith("benign")]
        print(f"{name:14s} over-refusal: " + " ".join(f"{k.split('|')[1][:6]}={fmt([r[k] for r in rows])}" for k in b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--alpha-rel", type=float, default=1.0)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--caps", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gcg", default=None, help="json {uid: suffixed prompt} from gcg.py")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    result = run(args)
    summarize(result)
    out = ROOT / "results_v2"; out.mkdir(exist_ok=True)
    tag = f"{args.model.replace('/', '__')}_L{args.layer}{args.tag}"
    (out / f"gen_{tag}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
