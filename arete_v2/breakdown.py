"""Gaussian location-shift model: oracle robust-z TPR/FPR as a function of prevalence pi and separation d."""
import json
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MAD_TO_SD = 1.4826
ZSTAR = 2.52


def mix_cdf(x, pi, d):
    return (1 - pi) * norm.cdf(x) + pi * norm.cdf(x - d)


def stats(pi, d):
    m = brentq(lambda x: mix_cdf(x, pi, d) - 0.5, -10, d + 10)
    mad = brentq(lambda r: mix_cdf(m + r, pi, d) - mix_cdf(m - r, pi, d) - 0.5, 1e-6, 20)
    return m, MAD_TO_SD * mad


def rates(pi, d, z=ZSTAR):
    m, s = stats(pi, d)
    t = m + z * s
    return 1 - norm.cdf(t), 1 - norm.cdf(t - d)


def pi_star(d, floor=0.05):
    """Largest prevalence at which oracle robust-z still keeps TPR above `floor`."""
    grid = np.linspace(0.001, 0.5, 500)
    ok = [p for p in grid if rates(p, d)[1] >= floor]
    return max(ok) if ok else 0.0


if __name__ == "__main__":
    out = {}
    print("d   pi:   0.05   0.10   0.20   0.30   0.50 | pi*(TPR>=5%)   FPR@0.05  FPR@0.30")
    for d in (2.0, 3.0, 3.5, 4.0, 5.0):
        row = [rates(p, d)[1] for p in (0.05, 0.1, 0.2, 0.3, 0.5)]
        print(f"{d:.1f}      " + "  ".join(f"{r*100:5.1f}" for r in row),
              f"| {pi_star(d):.3f}      {rates(0.05, d)[0]*100:.2f}   {rates(0.3, d)[0]*100:.2f}")
        out[str(d)] = dict(tpr=row, pi_star=pi_star(d))
    json.dump(out, open("results_v2/breakdown_gaussian.json", "w"), indent=1)
