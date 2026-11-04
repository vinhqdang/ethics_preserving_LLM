"""Real-prompt corpus for ARETE v2.

Harmful behaviours: HarmBench (text), AdvBench, JailbreakBench (JBB) harmful.
Benign / over-refusal: JBB benign (matched to harmful), XSTest safe, OR-Bench-hard.
Splits are disjoint by *behaviour* (the unwrapped request), so no attack wrapper of a
test behaviour is ever seen in training.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_v2" / "raw"
SPLIT_PATH = ROOT / "data_v2" / "splits.json"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/"
    "harmful_behaviors.csv"
)
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/"
    "behavior_datasets/harmbench_behaviors_text_all.csv"
)


@dataclass(frozen=True)
class Prompt:
    uid: str
    source: str
    text: str
    harmful: bool
    category: str = ""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _uid(source: str, text: str) -> str:
    return f"{source}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def _download(url: str, name: str) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / name
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "arete-v2"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_text(encoding="utf-8")


def _load_advbench() -> list[Prompt]:
    rows = csv.DictReader(io.StringIO(_download(ADVBENCH_URL, "advbench.csv")))
    return [Prompt(_uid("advbench", r["goal"]), "advbench", r["goal"], True) for r in rows]


def _load_harmbench() -> list[Prompt]:
    rows = csv.DictReader(io.StringIO(_download(HARMBENCH_URL, "harmbench.csv")))
    out = []
    for r in rows:
        if r.get("FunctionalCategory") != "standard":
            continue  # contextual / copyright behaviours need extra context fields
        out.append(
            Prompt(_uid("harmbench", r["Behavior"]), "harmbench", r["Behavior"], True,
                   r.get("SemanticCategory", ""))
        )
    return out


def _load_jbb() -> tuple[list[Prompt], list[Prompt]]:
    from datasets import load_dataset

    harm = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    ben = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    h = [Prompt(_uid("jbb", r["Goal"]), "jbb", r["Goal"], True, r.get("Category", "")) for r in harm]
    b = [Prompt(_uid("jbb_benign", r["Goal"]), "jbb_benign", r["Goal"], False, r.get("Category", ""))
         for r in ben]
    return h, b


def _load_xstest() -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset("Paul/XSTest", split="train")
    out = []
    for r in ds:
        if str(r["label"]).lower() == "safe":
            out.append(Prompt(_uid("xstest", r["prompt"]), "xstest", r["prompt"], False,
                              str(r.get("type", ""))))
    return out


def _load_orbench(limit: int = 400) -> list[Prompt]:
    from datasets import load_dataset

    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    out = [Prompt(_uid("orbench", r["prompt"]), "orbench", r["prompt"], False, r.get("category", ""))
           for r in ds]
    return out[:limit]


def _dedupe(items: Iterable[Prompt]) -> list[Prompt]:
    seen: set[str] = set()
    out: list[Prompt] = []
    for p in items:
        key = _norm(p.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def load_corpus() -> tuple[list[Prompt], list[Prompt]]:
    """Return (harmful, benign), de-duplicated across sources (JBB first)."""
    jbb_h, jbb_b = _load_jbb()
    harmful = _dedupe(jbb_h + _load_harmbench() + _load_advbench())
    benign = _dedupe(jbb_b + _load_xstest() + _load_orbench())
    return harmful, benign


def make_splits(seed: int = 0, frac=(0.5, 0.2, 0.3)) -> dict[str, dict[str, list[str]]]:
    """Disjoint-by-behaviour train/val/test split, stratified by source."""
    import random

    harmful, benign = load_corpus()
    rng = random.Random(seed)
    splits: dict[str, dict[str, list[str]]] = {k: {"harmful": [], "benign": []} for k in ("train", "val", "test")}
    for label, pool in (("harmful", harmful), ("benign", benign)):
        by_src: dict[str, list[Prompt]] = {}
        for p in pool:
            by_src.setdefault(p.source, []).append(p)
        for src_items in by_src.values():
            src_items = sorted(src_items, key=lambda p: p.uid)
            rng.shuffle(src_items)
            n = len(src_items)
            a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
            splits["train"][label] += [p.uid for p in src_items[:a]]
            splits["val"][label] += [p.uid for p in src_items[a:b]]
            splits["test"][label] += [p.uid for p in src_items[b:]]
    return splits


def build_and_save(seed: int = 0) -> dict:
    harmful, benign = load_corpus()
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "prompts": {p.uid: asdict(p) for p in harmful + benign},
        "splits": make_splits(seed),
    }
    SPLIT_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def load_split(name: str) -> tuple[list[Prompt], list[Prompt]]:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    table = {k: Prompt(**v) for k, v in payload["prompts"].items()}
    s = payload["splits"][name]
    return [table[u] for u in s["harmful"]], [table[u] for u in s["benign"]]


if __name__ == "__main__":
    payload = build_and_save()
    for name, s in payload["splits"].items():
        print(name, {k: len(v) for k, v in s.items()})
