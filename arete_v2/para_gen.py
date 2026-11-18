"""Undefended and always-on generations for TEST prompts wrapped with the paraphrased templates.

Same target, steer and decoding as fresh_gen.py; only the wrapper changes (para_templates.wrap_para).
A gate's output on a fired prompt equals the always-on output, so these two sets score any gate.
"""

from __future__ import annotations

import argparse
import json

from .data import ROOT, load_split
from .defenses import Cells
from .gen_eval import mean_norm, steer
from .geometry import refusal_direction
from .llm import generate, load_model
from .para_eval import FAMS
from .para_templates import PATH, wrap_para


def out_path(model: str):
    return ROOT / "results_v2" / f"gen_para_{model.replace('/', '__')}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha-rel", type=float, default=0.4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()
    table = json.loads(PATH.read_text(encoding="utf-8"))
    harm, ben = load_split("test")
    tr = Cells(args.model, args.layer, "train")
    rdir = refusal_direction(tr.acts[("harmful", "direct")], tr.acts[("benign", "direct")])
    alpha = args.alpha_rel * mean_norm(args.model, args.layer)
    path = out_path(args.model)
    blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "alpha": alpha, "requests": {"harmful": [p.text for p in harm], "benign": [p.text for p in ben]},
        "generations": {}}
    gens = blob["generations"]
    model, tok = load_model(args.model)
    for lab, items in (("harmful", harm), ("benign", ben)):
        for fam in FAMS:
            if fam == "direct" or f"always_on|{lab}|{fam}" in gens:
                continue
            texts = [wrap_para(fam, p.text, table) for p in items]
            gens[f"none|{lab}|{fam}"] = generate(model, tok, texts, None, args.tokens, args.bs)
            gens[f"always_on|{lab}|{fam}"] = generate(model, tok, texts, steer(args.layer, rdir, alpha),
                                                      args.tokens, args.bs)
            path.write_text(json.dumps(blob), encoding="utf-8")  # resumable
            print(lab, fam, "done", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
