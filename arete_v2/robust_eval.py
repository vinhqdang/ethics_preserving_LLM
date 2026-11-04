"""Non-transductive, multi-window evaluation of gating rules, at detection and generation level.

Addresses three review concerns:
  * transductive windows -> statistics are estimated on behaviours of half A and applied to half B;
  * a single lucky draw  -> ``--windows`` random A/B splits, mean and 95% percentile interval;
  * benign-rich traffic  -> per-framing harmful prevalence can differ (``--scenario``).
Generation-level ASR is composed from judged labels already computed: a fired prompt takes the label of
its always-on steered response (same steer, greedy decoding), an unfired one takes the undefended label.

Rules compared (all use the same probe score unless stated):
  shared      fixed threshold calibrated on VAL benign of seen framings (1% FPR)
  global_z    robust z over the whole window (k=1), threshold z* from VAL streams
  adaptive    per-cluster robust z (ours)
  family_z    per-family robust z using the true framing labels (oracle grouping)
  hybrid      shared (FPR/2) OR adaptive (FPR/2)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch

from .attacks import FAMILIES, FRESH_FAMILIES, SEEN_FAMILIES
from .data import ROOT, load_split
from .extract_multi import POSITIONS
from .lofo import MultiCells
from .recal import Config, fit_gate, robust_stats
from .recal_gen import fit_stream_model

M_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
DUAL_USE = ("direct", "fiction", "indirect", "past_tense", "academic", "code", "translation")
SCENARIOS = {
    # family -> harmful prevalence among that family's prompts in the window
    "uniform5": lambda f: 0.05,
    "realistic": lambda f: 0.05 if f in DUAL_USE else 0.5,
}


@dataclass
class Pool:
    """Per (label, family): score, cluster features, judged unsafe/refusal for none and always-on."""
    score: dict
    feats: dict
    unsafe_none: dict
    unsafe_on: dict
    refuse_none: dict
    refuse_on: dict


def _labels(path, fams) -> tuple[dict, dict, dict, dict]:
    lab = json.loads(path.read_text(encoding="utf-8"))["labels"]
    un, uo, rn, ro = {}, {}, {}, {}
    for l in ("harmful", "benign"):
        for f in fams:
            for key, dst_u, dst_r in ((f"none|{l}|{f}", un, rn), (f"always_on|{l}|{f}", uo, ro)):
                if key in lab:
                    dst_u[(l, f)] = np.array([s == "Unsafe" for s in lab[key]["safety"]])
                    dst_r[(l, f)] = np.array(lab[key]["refusal"], dtype=bool)
    return un, uo, rn, ro


def build_pool(model: str, fams, cfg: Config, score_fn) -> Pool:
    te = MultiCells(model, "test", fams)
    sl, sp = te.layers.index(cfg.score_layer), POSITIONS.index(cfg.score_view)
    cl, cp = te.layers.index(cfg.cluster_layer), POSITIONS.index(cfg.cluster_view)
    score, feats = {}, {}
    for k, a in te.acts.items():
        with torch.no_grad():
            score[k] = score_fn(a[:, sl, sp].float()).cpu().numpy()
        feats[k] = a[:, cl, cp].float().numpy()
    tag = model.replace("/", "__")
    un, uo, rn, ro = _labels(ROOT / "results_v2" / f"gen_{tag}_L14_full_judged.json", fams)
    fresh = ROOT / "results_v2" / f"gen_fresh_{tag}_judged.json"
    if fresh.exists():
        for d, e in zip((un, uo, rn, ro), _labels(fresh, fams)):
            d.update(e)
    return Pool(score, feats, un, uo, rn, ro)


def _z(scores, center, scale):
    return (scores - center) / max(scale, 1e-6)


def one_window(pool: Pool, fams, scenario, rng, thr, thr_h, zs, zs_h, fb, cfg) -> dict:
    n_h = len(next(v for (l, _), v in pool.score.items() if l == "harmful"))
    n_b = len(next(v for (l, _), v in pool.score.items() if l == "benign"))
    in_a_h, in_a_b = rng.random(n_h) < 0.5, rng.random(n_b) < 0.5
    # ---- traffic window from half A
    s_w, f_w, fam_w = [], [], []
    for f in fams:
        b_idx = np.flatnonzero(in_a_b)
        p = scenario(f)
        k = int(round(p / (1 - p) * len(b_idx)))
        h_idx = rng.choice(np.flatnonzero(in_a_h), size=min(k, int(in_a_h.sum())), replace=False)
        for l, idx in (("benign", b_idx), ("harmful", h_idx)):
            s_w.append(pool.score[(l, f)][idx]); f_w.append(pool.feats[(l, f)][idx]); fam_w += [f] * len(idx)
    s_w, f_w, fam_w = np.concatenate(s_w), np.concatenate(f_w), np.array(fam_w)
    sm = fit_stream_model(s_w, f_w, cfg)
    g_stats = robust_stats(s_w)
    fam_stats = {f: robust_stats(s_w[fam_w == f]) for f in fams}
    # ---- apply to half B
    out = {}
    for l, in_a in (("harmful", in_a_h), ("benign", in_a_b)):
        for f in fams:
            idx = np.flatnonzero(~in_a)
            s = pool.score[(l, f)][idx]
            c = sm.assign(pool.feats[(l, f)][idx])
            cz = np.array([_z(si, *(sm.stats[ci] if np.sum(sm.km.labels_ == ci) >= 30 else fb))
                           for si, ci in zip(s, c)])
            rules = {
                "shared": s > thr,
                "global_z": _z(s, *g_stats) > zs,
                "adaptive": cz > zs,
                "family_z": _z(s, *fam_stats[f]) > zs,
                "hybrid": (s > thr_h) | (cz > zs_h),
            }
            for r, fire in rules.items():
                d = out.setdefault(f"{r}|{l}|{f}", {})
                d["fire"] = float(fire.mean())
                if (l, f) in pool.unsafe_none:
                    un, uo = pool.unsafe_none[(l, f)][idx], pool.unsafe_on[(l, f)][idx]
                    rn, ro = pool.refuse_none[(l, f)][idx], pool.refuse_on[(l, f)][idx]
                    d["asr"] = float(np.where(fire, uo, un).mean())
                    d["refusal"] = float(np.where(fire, ro, rn).mean())
    for f in fams:  # undefended reference on the same half B
        idx_h, idx_b = np.flatnonzero(~in_a_h), np.flatnonzero(~in_a_b)
        if ("harmful", f) in pool.unsafe_none:
            out[f"none|harmful|{f}"] = {"fire": 0.0, "asr": float(pool.unsafe_none[("harmful", f)][idx_h].mean())}
            out[f"none|benign|{f}"] = {"fire": 0.0, "refusal": float(pool.refuse_none[("benign", f)][idx_b].mean())}
    return out


def evaluate(model: str, scenario_name: str, windows: int, fpr: float, include_fresh: bool) -> dict:
    cfg = Config(score_view="mean", cluster_layer=14)
    fams = tuple(FAMILIES) + (tuple(FRESH_FAMILIES) if include_fresh else ())
    tr = MultiCells(model, "train", SEEN_FAMILIES)
    va = MultiCells(model, "val", SEEN_FAMILIES)
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr)
    _, thr_h, zs_h, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr / 2)
    pool = build_pool(model, fams, cfg, score)
    rng = np.random.default_rng(0)
    runs = [one_window(pool, fams, SCENARIOS[scenario_name], rng, thr, thr_h, zs, zs_h, fb, cfg)
            for _ in range(windows)]
    summary = {}
    for key in runs[0]:
        for metric in runs[0][key]:
            v = np.array([r[key][metric] for r in runs])
            summary.setdefault(key, {})[metric] = [float(v.mean()), float(np.percentile(v, 2.5)),
                                                   float(np.percentile(v, 97.5))]
    return {"scenario": scenario_name, "windows": windows, "families": list(fams), "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=M_DEFAULT)
    ap.add_argument("--scenario", default="uniform5", choices=list(SCENARIOS))
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    res = evaluate(args.model, args.scenario, args.windows, args.fpr, args.fresh)
    tag = f"{args.model.replace('/', '__')}_{args.scenario}{'_fresh' if args.fresh else ''}"
    (ROOT / "results_v2" / f"robust_{tag}.json").write_text(json.dumps(res, indent=1))
    s = res["summary"]
    rules = ("shared", "global_z", "family_z", "adaptive", "hybrid")
    print(f"{'family':18s}" + "".join(f"{r:>18s}" for r in rules) + "   [harmful fire% | ASR%]")
    for f in res["families"]:
        row = []
        for r in rules:
            d = s[f"{r}|harmful|{f}"]
            asr = f"{d['asr'][0] * 100:4.0f}" if "asr" in d else "  --"
            row.append(f"{d['fire'][0] * 100:6.0f} |{asr}")
        print(f"{f:18s}" + "".join(f"{x:>18s}" for x in row))
    print("benign fire% (FPR):")
    for f in res["families"]:
        print(f"{f:18s}" + "".join(f"{s[f'{r}|benign|{f}']['fire'][0] * 100:18.1f}" for r in rules))


if __name__ == "__main__":
    main()
