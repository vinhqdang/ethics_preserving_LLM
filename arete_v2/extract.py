"""Cache last-token residual activations for every (split, label, family) cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .attacks import FAMILIES, wrap
from .data import ROOT, load_split
from .llm import extract_acts, load_model

CACHE = ROOT / "cache_v2"


def cache_dir(model_name: str, layer: int) -> Path:
    return CACHE / f"{model_name.replace('/', '__')}_L{layer}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    out = cache_dir(args.model, args.layer)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model, torch.bfloat16 if args.bf16 else torch.float16)
    for split in ("train", "val", "test"):
        harmful, benign = load_split(split)
        for label, items in (("harmful", harmful), ("benign", benign)):
            for fam in FAMILIES:
                path = out / f"{split}_{label}_{fam}.pt"
                if path.exists():
                    continue
                texts = [wrap(fam, p.text) for p in items]
                acts = extract_acts(model, tok, texts, args.layer, args.bs)
                torch.save({"acts": acts, "uids": [p.uid for p in items]}, path)
                print(split, label, fam, tuple(acts.shape), flush=True)
    (out / "meta.json").write_text(json.dumps({"model": args.model, "layer": args.layer}))


if __name__ == "__main__":
    main()
