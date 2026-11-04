"""Generate the manuscript's LaTeX tables from saved results (and cached activations).

Writes manuscript/tables/*.tex. Every number in the paper's tables comes from here.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from .attacks import FAMILIES, SEEN_FAMILIES
from .data import ROOT
from .defenses import calibrate
from .lofo import MultiCells, fit_probe
from .recal import Config, fit_gate, run_stream

M = "Qwen/Qwen2.5-1.5B-Instruct"
TAG = M.replace("/", "__")
RES = ROOT / "results_v2"
OUT = ROOT / "manuscript" / "tables"
FAM_SHORT = {"direct": "Direct", "persona": "Persona", "refusal_suppression": "Ref.\\,supp.",
             "prefix_injection": "Prefix inj.", "fiction": "Fiction", "indirect": "Indirect",
             "base64": "Base64", "leetspeak": "Leetspeak"}
UNSEEN = ("fiction", "indirect", "base64", "leetspeak")


def _hdr(cols: list[str]) -> str:
    return " & ".join(cols) + " \\\\"


def _fam_header() -> str:
    seen = [FAM_SHORT[f] for f in FAMILIES if f not in UNSEEN]
    unseen = [FAM_SHORT[f] for f in UNSEEN]
    return (f" & \\multicolumn{{{len(seen)}}}{{c}}{{Seen framings}} & "
            f"\\multicolumn{{{len(unseen)}}}{{c}}{{Unseen framings}} \\\\\n"
            f"\\cmidrule(lr){{2-{1 + len(seen)}}}\\cmidrule(lr){{{2 + len(seen)}-{1 + len(FAMILIES)}}}\n"
            + _hdr(["Gate"] + seen + unseen))


def table_detection() -> None:
    det = json.loads((RES / f"detect_{TAG}_L14.json").read_text())
    mv = json.loads((RES / f"mv_detect_{TAG}.json").read_text())
    rows = [("CAST", lambda f: det[f"cast|{f}"]["tpr"][0], lambda f: det[f"cast|{f}"]["auc"][0]),
            ("MLP probe", lambda f: det[f"mlp_probe|{f}"]["tpr"][0], lambda f: det[f"mlp_probe|{f}"]["auc"][0]),
            ("ARETE (v1 geometry)", lambda f: det[f"arete|{f}"]["tpr"][0], lambda f: det[f"arete|{f}"]["auc"][0]),
            ("Probe, last token", lambda f: mv[f"probe_last|{f}"]["tpr"], lambda f: mv[f"probe_last|{f}"]["auc"]),
            ("Probe, user tokens (mean)", lambda f: mv[f"probe_mean|{f}"]["tpr"], lambda f: mv[f"probe_mean|{f}"]["auc"]),
            ("Multi-view union", lambda f: mv[f"multiview|{f}"]["tpr"], lambda f: mv[f"multiview|{f}"]["auc"])]
    lines = ["\\begin{tabular}{l" + "c" * len(FAMILIES) + "}", "\\toprule", _fam_header(), "\\midrule"]
    for name, tpr, auc in rows:
        cells = [f"{100 * tpr(f):.0f}\\,{{\\scriptsize({auc(f):.2f})}}" for f in FAMILIES]
        lines.append(_hdr([name] + cells))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "detection.tex").write_text("\n".join(lines))


def table_oracle() -> None:
    tr = MultiCells(M, "train", SEEN_FAMILIES)
    va = MultiCells(M, "val", SEEN_FAMILIES)
    te = MultiCells(M, "test", tuple(FAMILIES))
    li = tr.layers.index(14)
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             _hdr(["Readout", "Framing", "AUROC", "TPR (shared)", "TPR (oracle)"]), "\\midrule"]
    for pi, view in ((0, "Last token"), (2, "User mean")):
        sc = fit_probe(tr, SEEN_FAMILIES, li, pi)
        with torch.no_grad():
            thr = calibrate(sc(va.get("benign", SEEN_FAMILIES, li, pi)).cpu(), 0.01)
            for k, fam in enumerate(UNSEEN):
                sh = sc(te.acts[("harmful", fam)][:, li, pi].float()).cpu()
                sb = sc(te.acts[("benign", fam)][:, li, pi].float()).cpu()
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(np.r_[np.ones(len(sh)), np.zeros(len(sb))], torch.cat([sh, sb]).numpy())
                othr = calibrate(sb, 0.01)
                lines.append(_hdr([view if k == 0 else "", FAM_SHORT[fam], f"{auc:.2f}",
                                   f"{100 * (sh > thr).float().mean():.1f}",
                                   f"{100 * (sh > othr).float().mean():.1f}"]))
        lines.append("\\midrule" if pi == 0 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "oracle.tex").write_text("\n".join(lines))


def table_recal(draws: int = 20) -> None:
    cfg = Config(score_view="mean", cluster_layer=14)
    tr = MultiCells(M, "train", SEEN_FAMILIES)
    va = MultiCells(M, "val", SEEN_FAMILIES)
    te = MultiCells(M, "test", tuple(FAMILIES))
    seen = list(SEEN_FAMILIES)
    sc, thr, zs, fb = fit_gate(tr, va, seen, cfg, 0.01)
    _, thr2, zs2, _ = fit_gate(tr, va, seen, cfg, 0.005)
    res = {}
    for pi in (0.01, 0.05, 0.10, 0.20):
        res[pi] = run_stream(sc, thr, zs, fb, te, tuple(FAMILIES), pi, draws, cfg, hybrid=(thr2, zs2))
    (RES / f"recal_tables_{TAG}.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))
    seen_f = [f for f in FAMILIES if f not in UNSEEN]
    n_s, n_u = len(seen_f), len(UNSEEN)
    header = (f" & & \\multicolumn{{{n_s}}}{{c}}{{Seen framings (TPR)}} & "
              f"\\multicolumn{{{n_u}}}{{c}}{{Unseen framings (TPR)}} & \\\\\n"
              f"\\cmidrule(lr){{3-{2 + n_s}}}\\cmidrule(lr){{{3 + n_s}-{2 + n_s + n_u}}}\n"
              + _hdr(["$\\pi$", "Rule"] + [FAM_SHORT[f] for f in seen_f]
                     + [FAM_SHORT[f] for f in UNSEEN] + ["FPR"]))
    lines = ["\\begin{tabular}{ll" + "c" * len(FAMILIES) + "c}", "\\toprule", header, "\\midrule"]
    for pi, r in res.items():
        for j, rule in enumerate(("shared", "adaptive", "hybrid")):
            cells = [f"{100 * r[f'{rule}|{f}']['tpr']:.0f}" for f in FAMILIES]
            fpr = np.mean([r[f"{rule}|{f}"]["fpr"] for f in FAMILIES])
            lines.append(_hdr([f"{int(pi * 100)}\\%" if j == 0 else "", rule.capitalize()] + cells
                              + [f"{100 * fpr:.1f}"]))
        lines.append("\\midrule" if pi != 0.20 else "\\bottomrule")
    lines.append("\\end{tabular}")
    (OUT / "recal.tex").write_text("\n".join(lines))


def _judged() -> dict:
    m = json.loads((RES / f"gen_{TAG}_L14_full_judged.json").read_text(encoding="utf-8"))["metrics"]
    for extra in (f"gen_mv_{TAG}_judged.json", f"gen_adaptive_{TAG}_judged.json"):
        p = RES / extra
        if p.exists():
            em = json.loads(p.read_text(encoding="utf-8"))["metrics"]
            m.update({k: v for k, v in em.items() if not k.startswith("none|")})
    return m


def table_generation() -> None:
    m = _judged()
    rows = [("None", "none"), ("Always-on steer", "always_on"), ("CAST", "cast"),
            ("Linear probe (last)", "linear_probe"), ("ARETE", "arete"),
            ("Multi-view union", "multiview"), ("Adaptive recal.\\ (ours)", "adaptive")]

    def val(prefix: str, lab: str, fam: str, metric: str):
        ks = [k for k in m if k.startswith(prefix + "|") and k.endswith(f"|{lab}|{fam}")]
        return float(np.mean([m[k][metric] for k in ks])) if ks else None

    fmt = lambda v: "--" if v is None else f"{100 * v:.1f}"
    seen_f = [f for f in FAMILIES if f not in UNSEEN]
    header = (f" & \\multicolumn{{{len(seen_f)}}}{{c}}{{Seen framings}} & "
              f"\\multicolumn{{{len(UNSEEN)}}}{{c}}{{Unseen framings}} & & \\\\\n"
              f"\\cmidrule(lr){{2-{1 + len(seen_f)}}}\\cmidrule(lr){{{2 + len(seen_f)}-{1 + len(FAMILIES)}}}\n"
              + _hdr(["Defense"] + [FAM_SHORT[f] for f in FAMILIES] + ["GCG", "Over-ref."]))
    lines = ["\\begin{tabular}{l" + "c" * (len(FAMILIES) + 2) + "}", "\\toprule", header, "\\midrule"]
    for name, key in rows:
        cells = [fmt(val(key, "harmful", f, "asr_strict")) for f in FAMILIES]
        cells += [fmt(val(key, "harmful", "gcg", "asr_strict")),
                  fmt(val(key, "benign", "direct", "over_refusal"))]
        lines.append(_hdr([name] + cells))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "generation.tex").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table_detection()
    table_oracle()
    table_recal()
    table_generation()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
