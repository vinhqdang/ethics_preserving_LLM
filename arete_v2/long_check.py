"""Robustness check for judged ASR: regenerate harmful TEST prompts with 256 new tokens.

The main tables judge 96-token responses, which mostly measures the onset of compliance. Here the
undefended and always-on responses of the key framings are regenerated at 256 tokens and re-judged,
to check that the gap between them (which drives every composed gate ASR) survives longer outputs.
"""

from __future__ import annotations

import json

from .attacks import wrap_any
from .data import ROOT, load_split
from .defenses import Cells
from .gen_eval import mean_norm, steer
from .geometry import refusal_direction
from .llm import generate, load_model

M = "Qwen/Qwen2.5-1.5B-Instruct"
FAMS = ("direct", "persona", "prefix_injection", "fiction", "indirect", "base64", "payload_split")
TOKENS = 256


def main() -> None:
    harm, ben = load_split("test")
    tr = Cells(M, 14, "train")
    rdir = refusal_direction(tr.acts[("harmful", "direct")], tr.acts[("benign", "direct")])
    alpha = 0.4 * mean_norm(M, 14)
    model, tok = load_model(M)
    gens = {}
    for fam in FAMS:
        texts = [wrap_any(fam, p.text) for p in harm]
        gens[f"none|harmful|{fam}"] = generate(model, tok, texts, None, TOKENS, 24)
        gens[f"always_on|harmful|{fam}"] = generate(model, tok, texts, steer(14, rdir, alpha), TOKENS, 24)
        print(fam, "done", flush=True)
    out = ROOT / "results_v2" / f"gen_long_{M.replace('/', '__')}.json"
    out.write_text(json.dumps({"alpha": alpha, "tokens": TOKENS,
                               "requests": {"harmful": [p.text for p in harm], "benign": [p.text for p in ben]},
                               "generations": gens}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
