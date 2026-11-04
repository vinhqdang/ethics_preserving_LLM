"""Generation-level evaluation of the label-free adaptive gate.

Deployment simulation: per-cluster robust statistics are estimated once from an unlabeled traffic
window (all TEST benign prompts + harmful at prevalence pi), then frozen and applied to every TEST
prompt via its nearest cluster centroid. Fired prompts are regenerated with the shared refusal steer;
the rest reuse the undefended generations. Writes a generations file for judge.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .attacks import FAMILIES, SEEN_FAMILIES, wrap
from .data import ROOT, load_split
from .defenses import Cells
from .extract_multi import POSITIONS
from .gen_eval import steer
from .geometry import refusal_direction
from .llm import generate, load_model
from .lofo import MultiCells
from .recal import K_RANGE, Config, build_stream, fit_gate, robust_stats


@dataclass(frozen=True)
class StreamModel:
    mean: np.ndarray
    std: np.ndarray
    pca: PCA
    km: KMeans
    stats: tuple[tuple[float, float], ...]

    def assign(self, feats: np.ndarray) -> np.ndarray:
        return self.km.predict(self.pca.transform((feats - self.mean) / self.std))


def fit_stream_model(scores: np.ndarray, feats: np.ndarray, cfg: Config, seed: int = 0) -> StreamModel:
    mean, std = feats.mean(0), feats.std(0) + 1e-6
    pca = PCA(n_components=cfg.pca_dim, random_state=seed).fit((feats - mean) / std)
    z = pca.transform((feats - mean) / std)
    best, best_km = -1.0, None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(z)
        s = silhouette_score(z, km.labels_, sample_size=min(2000, len(z)), random_state=seed)
        if s > best:
            best, best_km = s, km
    stats = tuple(robust_stats(scores[best_km.labels_ == c]) for c in range(best_km.n_clusters))
    return StreamModel(mean, std, pca, best_km, stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gen-json", required=True, help="gen_eval output with none|* generations")
    ap.add_argument("--pi", type=float, default=0.05)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=96)
    args = ap.parse_args()

    cfg = Config(score_view="mean", cluster_layer=14)
    tr = MultiCells(args.model, "train", SEEN_FAMILIES)
    va = MultiCells(args.model, "val", SEEN_FAMILIES)
    te = MultiCells(args.model, "test", tuple(FAMILIES))
    score, _, zs, _ = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, args.fpr)

    # 1. traffic window -> frozen per-cluster statistics
    x, feats, _, _ = build_stream(te, tuple(FAMILIES), args.pi, np.random.default_rng(0), cfg)
    with torch.no_grad():
        sm = fit_stream_model(score(x).cpu().numpy(), feats, cfg)
    print("clusters:", sm.km.n_clusters, flush=True)

    # 2. apply to every TEST prompt
    sl, sp = te.layers.index(cfg.score_layer), POSITIONS.index(cfg.score_view)
    cl, cp = te.layers.index(cfg.cluster_layer), POSITIONS.index(cfg.cluster_view)
    src = json.loads(Path(args.gen_json).read_text(encoding="utf-8"))
    harm, ben = load_split("test")
    trc = Cells(args.model, cfg.score_layer, "train")
    rdir = refusal_direction(trc.acts[("harmful", "direct")], trc.acts[("benign", "direct")])
    model, tok = load_model(args.model)
    gens = {k: v for k, v in src["generations"].items() if k.startswith("none|")}
    fire_rates = {}
    for lab, items in (("harmful", harm), ("benign", ben)):
        for fam in FAMILIES:
            acts = te.acts[(lab, fam)]
            with torch.no_grad():
                s = score(acts[:, sl, sp].float()).cpu().numpy()
            c = sm.assign(acts[:, cl, cp].float().numpy())
            z = np.array([(si - sm.stats[ci][0]) / max(sm.stats[ci][1], 1e-6) for si, ci in zip(s, c)])
            fired = z > zs
            fire_rates[f"{lab}|{fam}"] = float(fired.mean())
            texts = [wrap(fam, p.text) for p in items]
            outs = list(gens[f"none|{lab}|{fam}"])
            idx = np.flatnonzero(fired)
            if len(idx):
                sub = generate(model, tok, [texts[i] for i in idx],
                               steer(cfg.score_layer, rdir, src["alpha"]), args.tokens, args.bs)
                for i, o in zip(idx, sub):
                    outs[i] = o
            gens[f"adaptive|{lab}|{fam}"] = outs
            print(lab, fam, "fire", round(fire_rates[f"{lab}|{fam}"], 3), flush=True)
    res = {"alpha": src["alpha"], "requests": src["requests"], "gcg_requests": src.get("gcg_requests"),
           "generations": gens, "fire": fire_rates, "z_star": zs, "pi": args.pi}
    out = ROOT / "results_v2" / f"gen_adaptive_{args.model.replace('/', '__')}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
