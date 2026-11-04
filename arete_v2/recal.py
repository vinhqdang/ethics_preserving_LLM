"""Label-free, framing-adaptive recalibration of an activation gate.

Deployment model: prompts arrive as an unlabeled stream whose harmful prevalence is small (pi).
  1. Cluster the stream by *framing* features (early-layer mean-pooled activations, PCA + k-means,
     k by silhouette) - no family labels are used.
  2. Inside each cluster, robustly standardise the gate's score with median / MAD; both are
     insensitive to a few-percent harmful contamination.
  3. Fire when the robust z exceeds z*, where z* is fixed once on VAL benign (seen framings) so the
     benign false-positive rate there equals the budget.
Selection (score view, clustering layer) is done by leave-one-framing-out on VAL (``--select``);
TEST is then evaluated once with the selected configuration (``--test``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .attacks import FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .defenses import calibrate
from .extract_multi import POSITIONS
from .lofo import SEEN_WRAPPERS, MultiCells, fit_probe

MAD_TO_SD = 1.4826
K_RANGE = range(2, 13)


@dataclass(frozen=True)
class Config:
    score_layer: int = 14
    score_view: str = "mean"
    cluster_layer: int = 7
    cluster_view: str = "mean"
    pca_dim: int = 32


def robust_z(scores: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (scores - center) / max(scale, 1e-6)


def robust_stats(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, float(MAD_TO_SD * np.median(np.abs(x - med)))


def z_star(benign_scores: np.ndarray, fpr: float) -> float:
    """Threshold on the robust z of benign scores so that benign FPR = budget (fit once on VAL)."""
    c, s = robust_stats(benign_scores)
    return float(calibrate(torch.tensor(robust_z(benign_scores, c, s)), fpr))


def cluster(features: np.ndarray, pca_dim: int, seed: int = 0) -> np.ndarray:
    z = PCA(n_components=min(pca_dim, features.shape[1]), random_state=seed).fit_transform(
        (features - features.mean(0)) / (features.std(0) + 1e-6))
    best, best_lab = -1.0, np.zeros(len(z), dtype=int)
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(z)
        s = silhouette_score(z, lab, sample_size=min(2000, len(z)), random_state=seed)
        if s > best:
            best, best_lab = s, lab
    return best_lab


def adaptive_fire(scores: np.ndarray, labels: np.ndarray, zs: float, min_size: int = 30,
                  fallback: tuple[float, float] | None = None) -> np.ndarray:
    fire = np.zeros(len(scores), dtype=bool)
    for c in np.unique(labels):
        m = labels == c
        center, scale = robust_stats(scores[m]) if m.sum() >= min_size else fallback
        fire[m] = robust_z(scores[m], center, scale) > zs
    return fire


# ------------------------------------------------------------------ stream simulation
def build_stream(cells: MultiCells, fams, pi: float, rng: np.random.Generator, cfg: Config):
    """All benign prompts of every family + harmful prompts subsampled to prevalence pi per family."""
    sl, sp = cells.layers.index(cfg.score_layer), POSITIONS.index(cfg.score_view)
    cl, cp = cells.layers.index(cfg.cluster_layer), POSITIONS.index(cfg.cluster_view)
    rows = []
    for fam in fams:
        b = cells.acts[("benign", fam)]
        h = cells.acts[("harmful", fam)]
        n_h = max(1, int(round(pi / (1 - pi) * len(b))))
        idx = rng.choice(len(h), size=min(n_h, len(h)), replace=False)
        for acts, y in ((b, 0), (h[idx], 1)):
            rows.append((acts[:, sl, sp].float(), acts[:, cl, cp].float().numpy(),
                         np.full(len(acts), y), np.array([fam] * len(acts))))
    return (torch.cat([r[0] for r in rows]), np.concatenate([r[1] for r in rows]),
            np.concatenate([r[2] for r in rows]), np.concatenate([r[3] for r in rows]))


def run_stream(score_fn, thr_shared: float, zs: float, fallback, cells, fams, pi, draws, cfg,
               seed=0) -> dict:
    rng = np.random.default_rng(seed)
    acc: dict[str, dict[str, list[float]]] = {}
    for _ in range(draws):
        x, feats, y, fam = build_stream(cells, fams, pi, rng, cfg)
        with torch.no_grad():
            s = score_fn(x).cpu().numpy()
        shared = s > thr_shared
        adapt = adaptive_fire(s, cluster(feats, cfg.pca_dim), zs, fallback=fallback)
        for f in fams:
            m = fam == f
            for name, fire in (("shared", shared), ("adaptive", adapt)):
                d = acc.setdefault(f"{name}|{f}", {"tpr": [], "fpr": []})
                d["tpr"].append(float(fire[m & (y == 1)].mean()))
                d["fpr"].append(float(fire[m & (y == 0)].mean()))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}


def z_star_stream(score_fn, va: MultiCells, fams, cfg: Config, fpr: float, pi: float,
                  draws: int = 5, seed: int = 123) -> float:
    """z* such that the *clustered, robust-standardised* benign FPR on VAL streams of the given
    (seen) framings equals the budget - accounts for within-cluster dispersion."""
    rng = np.random.default_rng(seed)
    zs_benign = []
    for _ in range(draws):
        x, feats, y, _ = build_stream(va, fams, pi, rng, cfg)
        with torch.no_grad():
            s = score_fn(x).cpu().numpy()
        lab = cluster(feats, cfg.pca_dim)
        for c in np.unique(lab):
            m = lab == c
            center, scale = robust_stats(s[m])
            zs_benign.append(robust_z(s[m & (y == 0)], center, scale))
    return float(calibrate(torch.tensor(np.concatenate(zs_benign)), fpr))


def fit_gate(tr: MultiCells, va: MultiCells, fams, cfg: Config, fpr: float, pi: float = 0.05):
    li, pi_ = tr.layers.index(cfg.score_layer), POSITIONS.index(cfg.score_view)
    score = fit_probe(tr, fams, li, pi_)
    with torch.no_grad():
        ben = score(va.get("benign", fams, li, pi_)).cpu().numpy()
    zs = z_star_stream(score, va, fams, cfg, fpr, pi)
    return score, float(calibrate(torch.tensor(ben), fpr)), zs, robust_stats(ben)


def select(model: str, fpr: float, pi: float, draws: int) -> dict:
    fams_all = ("direct",) + SEEN_WRAPPERS
    tr, va = MultiCells(model, "train", fams_all), MultiCells(model, "val", fams_all)
    out = {}
    for view in POSITIONS:
        for cl in tr.layers:
            cfg = Config(score_view=view, cluster_layer=cl)
            folds = []
            for held in SEEN_WRAPPERS:
                seen = ["direct"] + [w for w in SEEN_WRAPPERS if w != held]
                score, thr, zs, fb = fit_gate(tr, va, seen, cfg, fpr)
                r = run_stream(score, thr, zs, fb, va, fams_all, pi, draws, cfg)
                folds.append((r[f"shared|{held}"], r[f"adaptive|{held}"],
                              np.mean([r[f"adaptive|{f}"]["fpr"] for f in fams_all])))
            key = f"{view}|clusterL{cl}"
            out[key] = {"shared_tpr": float(np.mean([f[0]["tpr"] for f in folds])),
                        "adaptive_tpr": float(np.mean([f[1]["tpr"] for f in folds])),
                        "adaptive_fpr_held": float(np.mean([f[1]["fpr"] for f in folds])),
                        "adaptive_fpr_all": float(np.mean([f[2] for f in folds]))}
            print(key, {k: round(v, 3) for k, v in out[key].items()}, flush=True)
    return out


def test(model: str, cfg: Config, fpr: float, pis, draws: int) -> dict:
    tr = MultiCells(model, "train", SEEN_FAMILIES)
    va = MultiCells(model, "val", SEEN_FAMILIES)
    te = MultiCells(model, "test", tuple(FAMILIES))
    score, thr, zs, fb = fit_gate(tr, va, list(SEEN_FAMILIES), cfg, fpr)
    out = {"config": cfg.__dict__, "z_star": zs}
    for pi in pis:
        out[f"pi={pi}"] = run_stream(score, thr, zs, fb, te, tuple(FAMILIES), pi, draws, cfg)
        print(f"pi={pi}")
        for f in FAMILIES:
            s, a = out[f"pi={pi}"][f"shared|{f}"], out[f"pi={pi}"][f"adaptive|{f}"]
            print(f"  {f:20s} shared TPR={s['tpr']:.3f} FPR={s['fpr']:.3f} | "
                  f"adaptive TPR={a['tpr']:.3f} FPR={a['fpr']:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--view", default="mean")
    ap.add_argument("--cluster-layer", type=int, default=7)
    args = ap.parse_args()
    dest = ROOT / "results_v2"
    tag = args.model.replace("/", "__")
    if args.select:
        res = select(args.model, args.fpr, 0.05, args.draws)
        (dest / f"recal_select_{tag}.json").write_text(json.dumps(res, indent=1))
    if args.test:
        cfg = Config(score_view=args.view, cluster_layer=args.cluster_layer)
        res = test(args.model, cfg, args.fpr, (0.01, 0.05), args.draws)
        (dest / f"recal_test_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
