"""Cache residual activations at several token positions and layers in one forward pass.

Positions (per prompt, chat-templated):
  * ``last``  - final template token (post-instruction; where refusal is decided; v1 default)
  * ``tinst`` - last token of the user's own text (where harmfulness is read out; Zhao et al., 2025)
  * ``mean``  - mean over the user's own tokens
Output per cell: {"acts": float16 [N, n_layers, 3, D], "layers": [...], "positions": [...]}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .attacks import FAMILIES, wrap
from .data import ROOT, load_split
from .llm import chat_text, layers_of, load_model

POSITIONS = ("last", "tinst", "mean")
SENTINEL = "<<U>>"


def multi_dir(model_name: str) -> Path:
    return ROOT / "cache_v2" / f"{model_name.replace('/', '__')}_multi"


def template_lengths(tok) -> tuple[int, int]:
    """Token counts of the chat template before and after the user's text."""
    before, after = chat_text(tok, SENTINEL).split(SENTINEL)
    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    return n(before), n(after)


@torch.no_grad()
def extract(model, tok, prompts: list[str], layers: list[int], bs: int) -> torch.Tensor:
    n_pre, n_post = template_lengths(tok)
    grabbed: dict[int, torch.Tensor] = {}
    hooks = [layers_of(model)[l].register_forward_hook(
        lambda _m, _a, out, l=l: grabbed.__setitem__(l, out[0] if isinstance(out, tuple) else out))
        for l in layers]
    chunks = []
    try:
        for i in range(0, len(prompts), bs):
            enc = tok([chat_text(tok, p) for p in prompts[i : i + bs]], return_tensors="pt",
                      padding=True).to("cuda")
            model(**enc)
            mask = enc["attention_mask"]
            seq = mask.shape[1]
            start = (seq - mask.sum(1)) + n_pre  # left padding: first user-text token
            end = torch.full_like(start, seq - n_post)  # exclusive end of user text
            pos = torch.arange(seq, device=mask.device)[None]
            user = ((pos >= start[:, None]) & (pos < end[:, None])).float()  # [B, T]
            per_layer = []
            for l in layers:
                h = grabbed[l].float()
                last = h[:, -1]
                tinst = h[torch.arange(h.shape[0]), end - 1]
                mean = (h * user[..., None]).sum(1) / user.sum(1, keepdim=True).clamp(min=1)
                per_layer.append(torch.stack([last, tinst, mean], 1))
            chunks.append(torch.stack(per_layer, 1).half().cpu())
    finally:
        for hk in hooks:
            hk.remove()
    return torch.cat(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21])
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    out = multi_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(args.model)
    for split in ("train", "val", "test"):
        harmful, benign = load_split(split)
        for label, items in (("harmful", harmful), ("benign", benign)):
            for fam in FAMILIES:
                path = out / f"{split}_{label}_{fam}.pt"
                if path.exists():
                    continue
                acts = extract(model, tok, [wrap(fam, p.text) for p in items], args.layers, args.bs)
                torch.save({"acts": acts, "layers": args.layers, "positions": list(POSITIONS),
                            "uids": [p.uid for p in items]}, path)
                print(split, label, fam, tuple(acts.shape), flush=True)


if __name__ == "__main__":
    main()
