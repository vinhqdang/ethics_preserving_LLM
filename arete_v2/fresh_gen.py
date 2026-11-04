"""Undefended and always-on generations for the pre-registered fresh framings (TEST split).

A gated defense's output on a fired prompt equals the always-on output (same steer, greedy decoding),
so these two sets suffice to score any gate's decisions on the fresh framings.
"""

from __future__ import annotations

import argparse
import json

import torch

from .attacks import FRESH_FAMILIES, wrap_any
from .data import ROOT, load_split
from .defenses import Cells
from .gen_eval import mean_norm, steer
from .geometry import refusal_direction
from .llm import generate, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha-rel", type=float, default=0.4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()
    harm, ben = load_split("test")
    tr = Cells(args.model, args.layer, "train")
    rdir = refusal_direction(tr.acts[("harmful", "direct")], tr.acts[("benign", "direct")])
    alpha = args.alpha_rel * mean_norm(args.model, args.layer)
    model, tok = load_model(args.model)
    gens = {}
    for lab, items in (("harmful", harm), ("benign", ben)):
        for fam in FRESH_FAMILIES:
            texts = [wrap_any(fam, p.text) for p in items]
            gens[f"none|{lab}|{fam}"] = generate(model, tok, texts, None, args.tokens, args.bs)
            gens[f"always_on|{lab}|{fam}"] = generate(model, tok, texts, steer(args.layer, rdir, alpha),
                                                      args.tokens, args.bs)
            print(lab, fam, "done", flush=True)
    out = ROOT / "results_v2" / f"gen_fresh_{args.model.replace('/', '__')}.json"
    out.write_text(json.dumps({"alpha": alpha, "requests": {"harmful": [p.text for p in harm],
                                                            "benign": [p.text for p in ben]},
                               "generations": gens}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
