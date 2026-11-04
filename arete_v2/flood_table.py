"""Render the flooding table (all flooded framings) from results_v2/flood_*.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("0.05", "0.2", "0.3", "0.5")
NAMES = {"fiction": "fiction", "code": "code", "academic": "academic",
         "indirect": "indirect", "past_tense": "past tense"}


def main(tag: str = "Qwen__Qwen2.5-1.5B-Instruct") -> None:
    data = json.loads((ROOT / "results_v2" / f"flood_{tag}.json").read_text())["flood"]
    head = "".join(f"& {p} " for p in LEVELS)
    lines = [
        r"\begin{tabular}{lc" + "c" * (2 * len(LEVELS)) + "}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{adaptive, $\pi=$} & \multicolumn{4}{c}{hybrid, $\pi=$} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"framing & shared " + head + head + r"\\",
        r"\midrule",
    ]
    for key, name in NAMES.items():
        rows = data[key]
        shared = sum(rows[p]["shared"] for p in rows) / len(rows)
        cells = [f"{rows[p]['adaptive'] * 100:.1f}" for p in LEVELS]
        cells += [f"{rows[p]['hybrid'] * 100:.1f}" for p in LEVELS]
        lines.append(f"{name} & {shared * 100:.1f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out = ROOT / "manuscript" / "tables" / "flood.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
