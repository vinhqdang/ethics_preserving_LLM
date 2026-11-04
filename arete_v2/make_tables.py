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


FRESH_SHORT = {"past_tense": "Past tense", "academic": "Academic", "code": "Code",
               "translation": "Translation", "payload_split": "Payload split"}
ALL_SHORT = {**FAM_SHORT, **FRESH_SHORT}
ROBUST_RULES = (("shared", "Shared"), ("adaptive", "Adaptive"), ("hybrid", "Hybrid"))
NO_ASR = ("base64",)  # judged ASR is a judge artefact for base64 (decoding echoes), see text


def _robust(scenario: str, source: str = "prereg") -> dict:
    """prereg: k<=12, 20 windows (pre-registered); k30: post-hoc k<=30 sensitivity, 10 windows."""
    if source == "k30":
        return json.loads((RES / "kmax30_posthoc.json").read_text())[scenario]
    return json.loads((RES / f"robust_{TAG}_{scenario}_fresh.json").read_text())["summary"]


def _group(fam: str) -> str:
    if fam in FRESH_SHORT:
        return "Fresh (pre-registered)"
    return "Unseen" if fam in UNSEEN else "Seen"


def table_robust(source: str = "prereg") -> None:
    """Harmful TPR (%) per framing under two traffic scenarios; benign FPR (%) averaged."""
    su, sr = _robust("uniform5", source), _robust("realistic", source)
    fams = list(FAMILIES) + list(FRESH_SHORT)
    cols = " & ".join([f"{n}" for _, n in ROBUST_RULES] * 2)
    lines = ["\\begin{tabular}{ll" + "c" * 6 + "}", "\\toprule",
             " & & \\multicolumn{3}{c}{Benign-rich traffic} & \\multicolumn{3}{c}{Realistic traffic} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
             f"Group & Framing & {cols} \\\\", "\\midrule"]
    last_group = None
    for f in fams:
        g = _group(f)
        if last_group is not None and g != last_group:
            lines.append("\\midrule")
        cells = [f"{100 * s[f'{r}|harmful|{f}']['fire'][0]:.0f}" for s in (su, sr) for r, _ in ROBUST_RULES]
        lines.append(_hdr([g if g != last_group else "", ALL_SHORT[f]] + cells))
        last_group = g
    lines.append("\\midrule")
    fpr = [f"{100 * np.mean([s[f'{r}|benign|{f}']['fire'][0] for f in fams]):.1f}"
           for s in (su, sr) for r, _ in ROBUST_RULES]
    lines.append(_hdr(["", "Benign FPR (mean)"] + fpr))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"robust_{source}.tex").write_text("\n".join(lines))


def table_ablation() -> None:
    su = _robust("uniform5", "k30")
    fams = ("fiction", "indirect", "past_tense", "academic", "payload_split")
    rules = (("shared", "Shared threshold"), ("global_z", "Global robust $z$ ($k{=}1$)"),
             ("family_z", "Per-family $z$ (oracle groups)"), ("adaptive", "Per-cluster $z$ (ours)"))
    lines = ["\\begin{tabular}{l" + "c" * (len(fams) + 1) + "}", "\\toprule",
             _hdr(["Rule"] + [ALL_SHORT[f] for f in fams] + ["FPR"]), "\\midrule"]
    all_f = list(FAMILIES) + list(FRESH_SHORT)
    for r, name in rules:
        cells = [f"{100 * su[f'{r}|harmful|{f}']['fire'][0]:.0f}" for f in fams]
        fpr = np.mean([su[f"{r}|benign|{f}"]["fire"][0] for f in all_f])
        lines.append(_hdr([name] + cells + [f"{100 * fpr:.1f}"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "ablation.tex").write_text("\n".join(lines))


def table_generation_robust(scenario: str = "realistic", source: str = "k30") -> None:
    """Composed judged ASR (%) on half B; mean [95% interval] over windows."""
    su = _robust(scenario, source)
    fams = [f for f in list(FAMILIES) + list(FRESH_SHORT) if f not in NO_ASR]
    rows = (("none", "No defense"), ("shared", "Shared threshold"), ("adaptive", "Adaptive (ours)"),
            ("hybrid", "Hybrid (ours)"))
    lines = ["\\begin{tabular}{l" + "c" * len(rows) + "}", "\\toprule",
             _hdr(["Framing"] + [n for _, n in rows]), "\\midrule"]
    for f in fams:
        cells = []
        for r, _ in rows:
            d = su.get(f"{r}|harmful|{f}", {})
            cells.append("--" if "asr" not in d else f"{100 * d['asr'][0]:.0f}\\,{{\\scriptsize[{100 * d['asr'][1]:.0f},{100 * d['asr'][2]:.0f}]}}")
        lines.append(_hdr([ALL_SHORT[f]] + cells))
    lines.append("\\midrule")
    ref = []
    for r, _ in rows:
        d = su.get(f"{r}|benign|direct", {})
        ref.append("--" if "refusal" not in d else f"{100 * d['refusal'][0]:.1f}")
    lines.append(_hdr(["Over-refusal (plain benign)"] + ref))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"generation_{scenario}_{source}.tex").write_text("\n".join(lines))


def table_fresh_oracle() -> None:
    fo = json.loads((RES / f"fresh_oracle_{TAG}.json").read_text())
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             _hdr(["Fresh framing", "AUROC", "TPR (shared)", "TPR (oracle)"]), "\\midrule"]
    for f in FRESH_SHORT:
        d = fo[f"mean|{f}"]
        lines.append(_hdr([FRESH_SHORT[f], f"{d['auc']:.2f}", f"{100 * d['shared']:.1f}", f"{100 * d['oracle']:.1f}"]))
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "fresh_oracle.tex").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table_detection()
    table_oracle()
    table_recal()
    table_generation()
    table_robust("prereg")
    table_robust("k30")
    table_ablation()
    table_generation_robust("realistic", "k30")
    table_generation_robust("realistic", "prereg")
    table_fresh_oracle()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()


PHI_TAG = "microsoft__Phi-3.5-mini-instruct"


def table_second_model() -> None:
    """Replication on Phi-3.5-mini with the configuration frozen from the 1.5B study."""
    def load(k: int, scen: str) -> dict:
        return json.loads((RES / f"robust_{PHI_TAG}_{scen}_k{k}.json").read_text())["summary"]

    s30, s12 = load(30, "uniform5"), load(12, "uniform5")
    diag = json.loads((RES / f"diag_{PHI_TAG}.json").read_text())
    fams = ["fiction", "indirect", "past_tense", "academic", "code", "translation", "payload_split"]
    lines = [r"\begin{tabular}{lcccccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{Recall (\%)} & \multicolumn{3}{c}{Judged ASR (\%)} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
             _hdr(["Framing", "AUROC", "Shared", "Oracle", r"Hyb.\,12", r"Hyb.\,30", "None", "Shared",
                   r"Hyb.\,30"]),
             r"\midrule"]
    for f in fams:
        d = diag[f"mean|{f}"]
        rec = lambda s, r: f"{100 * s[f'{r}|harmful|{f}']['fire'][0]:.0f}"
        asr = lambda s, r: f"{100 * s[f'{r}|harmful|{f}']['asr'][0]:.0f}"
        lines.append(_hdr([ALL_SHORT[f], f"{d['auc']:.2f}", rec(s30, "shared"), f"{100 * d['oracle_tpr']:.0f}",
                           rec(s12, "hybrid"), rec(s30, "hybrid"), asr(s30, "none"), asr(s30, "shared"),
                           asr(s30, "hybrid")]))
    all_f = list(FAMILIES) + list(FRESH_SHORT)
    fpr = lambda s, r: 100 * np.mean([s[f"{r}|benign|{f}"]["fire"][0] for f in all_f])
    lines.append(r"\midrule")
    lines.append(_hdr(["Benign FPR (mean)", "", f"{fpr(s30, 'shared'):.1f}", "", f"{fpr(s12, 'hybrid'):.1f}",
                       f"{fpr(s30, 'hybrid'):.1f}", "", "", ""]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "second_model.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    table_second_model()
