# Writing and citation review of `manuscript/main.tex` (v2, "The Gate Knows, the Threshold Doesn't")

Reviewed 2026-11-04. I did not edit the manuscript. I checked the numbers quoted in the prose against `manuscript/tables/*.tex` and the compiled `main.pdf` (10 pages, `article` class, `plainnat`).

Severity: **CRITICAL** (fix before submission), **HIGH** (a reviewer will notice), **MEDIUM**, **LOW**.

---

## 0. Top fixes in priority order

1. **CRITICAL: internal verification tags are printed in the bibliography.** `plainnat` renders the BibTeX `note` field, so the PDF reference list shows strings such as "[K].", "[V] Spotlight.", "[V] CLOSEST PRIOR WORK.", and "[V] arXiv/authors verified; [U] SaTML 2025 venue not confirmed". Remove every `note` field, or move these tags into BibTeX comments, before anyone sees the PDF.
2. **CRITICAL (format): the paper does not use the ACL template.** It uses `article` with custom geometry, numeric `plainnat`, and no `acl.sty`, no line numbers, and no review mode. Switch to `\usepackage[review]{acl}` with `acl_natbib.bst`, which gives author-year citations.
3. **HIGH: there is no standalone Limitations section.** ACL requires an unnumbered `\section*{Limitations}` after the conclusion. It does not count toward the page limit. At present Limitations is a paragraph inside Section 7, so it counts toward the limit and may fail the format check.
4. **HIGH: factual mismatches between the prose and the tables** (§1.3): "every gate keeps more than 80% recall on seen framings" (CAST does not); "five such gates" (the tables show six); "a remedy that needs no labels" (z* is fit on labelled validation streams); the adaptive rule beats the oracle on fiction without explanation; and generation-level indirect firing is 92% against 65% in the detection table.
5. **HIGH: a close prior work is not cited.** `farzam2026hiding` (FrameShield/ReDAct) treats framing as a nuisance factor to disentangle, which is the same intuition as ARETE. `schwarz2026entanglement` makes the opposite claim, that refitting thresholds does not help. Both are in the bib but uncited. Reviewers who know them will flag the omission.
6. **HIGH: the abstract has 296 words.** ACL asks for at most about 200. A replacement is given in §1.2.
7. **MEDIUM: five cited BibTeX entries use "and others"**, one DOI is malformed, and several venues are unverified (§3).

---

## 1. Narrative flow and story-telling

### 1.1 Overall arc

The paper does follow problem → diagnosis → remedy → limits, and the title makes the thesis memorable. The arc breaks in four places:

- **Intro enumeration is inconsistent.** It runs "The first finding… The second finding… The third contribution…" and then "we make four contributions". The fourth contribution (limits) never appears in the narrative paragraphs. Use one counting scheme throughout.
- **Section 4 is out of order.** "Ranking survives" relies on the user-token-mean probe before "Where to read" explains why that readout exists. "Projecting out framing" sits between them and breaks the line from diagnosis to remedy. Suggested order: (a) recall collapses; (b) where to read, kept short, which motivates the user-mean readout; (c) ranking survives and the oracle recovers recall; (d) projecting out framing does not help, which is the negative result and leads straight into "so fix the threshold, not the representation".
- **The method section reports a test-set result.** "On the test traffic window silhouette chose k=12" is a result placed in the method. Move it to Results or to an appendix.
- **Discussion repeats the introduction** ("Activation gates are usually evaluated by AUROC…"). It should add something new instead. Candidates: what a practitioner should change (report TPR at a fixed FPR per framing, and monitor the cluster-wise median), and why the result implies representation-invariance methods like ARETE are targeting the wrong failure.
- **Three different LOFO baselines appear with no table**: 54% (linear probe, projection rank 0), 69% (union), and 38% → 91% (user-mean, shared → adaptive at π = 5%). A reader cannot reconcile them. Add one small LOFO table, or one sentence explaining that the numbers come from different readouts and prevalence settings.

### 1.2 Five most important rewrites (replacement text)

**R1. Abstract (296 → about 190 words; fixes the "no labels" overclaim, adds the costs, removes stock phrasing).**

> Activation-gated jailbreak defenses read a language model's residual stream, decide whether a prompt is harmful, and only then intervene, for example by steering towards refusal. Their decision thresholds are set on the jailbreak framings seen during development. We evaluate six gates on real harmful and benign prompts wrapped in eight framing families, four of them held out. At a 1\% benign false-positive budget, no gate detects more than 17\% of harmful fiction or indirect-injection prompts, and none lowers the 89\% attack success rate of fiction prompts. Yet a probe on the mean of the user's tokens still separates harmful from benign prompts on these framings (AUROC 0.98--0.99), and re-thresholding it per framing would recover 79--81\% recall: the threshold has shifted, not the signal. We therefore cluster the unlabeled prompt stream by framing and standardise the gate score within each cluster using the median and MAD. The method needs no labels from the new framing. Selected by leave-one-framing-out validation and run once on the test split, it raises recall on fiction from 7\% to 87\% and on indirect injection from 17\% to 65\% at 5\% harmful prevalence. It also cuts fiction attack success from 89\% to 34\%, at the cost of a mean false-positive rate of 1.6\% and lower recall on plain requests. It fails when harmful prompts exceed about a fifth of the traffic and when the model cannot parse the framing.

**R2. Intro paragraph 3 (line 37). The current claim that every gate keeps >80% recall is false for CAST.**

> The first finding is a collapse. At the 1\% budget, every probe-based gate detects 83--97\% of harmful prompts on seen framings, and CAST already struggles on two of them. On fiction and indirect injection, every gate, including a geometric gate of our own that tries to factor framing out of the representation, detects at most 17\%. None of them lowers the 89\% attack success rate of fiction prompts.

**R3. Method paragraph (line 130). Dense; mixes configuration, a test result, and the hybrid rule.**

> Algorithm~\ref{alg:recal} gives the procedure. The gate score $s$ is the user-token-mean probe. The clustering features $\phi$ are the same mean-pooled user-token activations, projected onto 32 principal components, and $k$ is chosen by silhouette from $\{2,\dots,12\}$. Clusters with fewer than 30 prompts fall back to validation statistics. Because the median and MAD tolerate a minority of harmful prompts, each prompt is compared with the benign bulk of its own framing rather than with that of the training framings. The threshold $z^\star$ is set on validation streams that pass through the same clustering, so it already reflects within-cluster dispersion.
>
> \paragraph{Hybrid rule.} As a safeguard for regimes where traffic statistics are unreliable, the hybrid rule fires if either the shared or the adaptive threshold fires, each at half the budget.

Move the "silhouette chose k=12 on the test window" sentence to Results.

**R4. Generation paragraph (line 156). It packs about 12 facts into one paragraph and hides the transductive estimation.**

> Detection gains matter only if they change what the model says (Table~\ref{tab:generation}). Two reference points bracket the gates. The always-on steer lowers fiction ASR from 88.8\% to 28.6\% but refuses every plain benign request. The fixed-threshold gates never fire on fiction or indirect injection, so their ASR there equals the undefended model's.
>
> For the adaptive gate we estimate per-cluster statistics once from a traffic window that contains the test prompts (all benign prompts plus harmful prompts at 5\% prevalence), freeze them, and route each prompt to its nearest cluster. In this transductive setting the gate fires on 86\% of harmful fiction and 92\% of harmful indirect prompts, and on 0.9\% of benign prompts in those framings. Fiction ASR falls to 34.4\%, within six points of the always-on bound, and indirect ASR falls to 4.6\%. Over-refusal on plain benign requests rises from 59.1\% to 60.0\%. The high base rate reflects how often the 1.5B model refuses the borderline prompts of XSTest and OR-Bench without any defense.
>
> On seen framings the adaptive gate matches the best fixed-threshold gate (prefix injection: 12.0\% vs.\ 12.9--13.7\%). Base64 is unaffected because the gate cannot see it (Section~\ref{sec:diagnosis}), and leetspeak never succeeds against this model.

Also explain the gap between 92% here and 65% in Table 3, which comes from a single frozen window versus 20 resampled streams. State why GCG is reported as "--" for the adaptive gate.

**R5. Limitations: turn it into a standalone section and remove the GPU excuse.**

> \section*{Limitations}
> \textbf{Scale.} All experiments use one 1.5B-parameter model. Larger models decode base64, which we expect would turn that framing from a capability failure into a calibration failure that recalibration can address. We have not tested this.
> \textbf{Threat model.} Our framings are template-level. We do not evaluate attacks optimised against the gate \citep{bailey2026obfuscated}. An adaptive attacker could also target the clustering, or flood a cluster to contaminate its median and MAD, which the prevalence study shows disables the adaptive rule above about 20\% prevalence.
> \textbf{Deployment assumptions.} The method needs a window of traffic in which harmful prompts are a minority of each framing cluster. The generation experiment estimates cluster statistics transductively, on a window that contains the evaluated prompts.
> \textbf{Evaluation.} The judge is a model. On wrapped benign prompts it often labels confused non-compliance as refusal, so we report over-refusal only for plain benign requests. The GCG evaluation uses 20 behaviours and is indicative only.

Keep a short "Discussion" section in the main text for the interpretation paragraph.

### 1.3 Accuracy issues in the prose (checked against tables)

| Location | Problem | Fix |
|---|---|---|
| Abstract, "five such gates" | Table 1 has six rows: CAST, MLP, ARETE, last-token, user-mean, union. Setup lists (i)–(v) with (v) plural. | Say "six gates", or define what counts as a gate. |
| Abstract and Section 5 title, "needs no labels" | z* is the (1−ε) quantile of benign validation z-scores, so validation labels are used. | Say "no labels from the deployment stream or the new framing". |
| Intro l.37 | ">80% recall on seen framings": CAST gets 42/50/0. | See R2. |
| §4 l.95 | "benign FPR on these framings also falls to nearly zero": no table shows per-framing benign FPR under the shared threshold. | Add it to Table 1 or 2, or cite an appendix. |
| §4 l.105 | "exception that proves the rule" (cliché; base64 actually contradicts the rule's scope). Leetspeak is left unexplained: last-token AUROC is 0.95 yet the oracle TPR is only 17.8%, which weakens "high AUROC ⇒ recoverable". | Discuss leetspeak explicitly. High AUROC with a heavy-tailed benign distribution still gives low TPR at 1% FPR. |
| §6 Detection | Adaptive fiction recall (85–87%) *exceeds* the oracle (80.5%) at FPR 0.6–1.0%. "Close to the oracle" hides this, and a reviewer will ask how a label-free rule beats a per-framing oracle. | Explain the cause: k=12 sub-clusters give finer-than-framing thresholds, and/or the populations differ (subsampled harmful set, 20 draws). |
| §6 Generation | 92% indirect firing vs 65–70% in Table 3. | Explain (R4). |
| Table 4 | No row for the same user-mean probe with a shared threshold, which is the natural ablation for "only the decision rule differs". The MLP row is also missing. | Add the shared user-mean row. |
| Table 4 caption, "Gated rows average three seeds" | Decoding is greedy, so it is unclear what the seed controls. | State it (probe training / stream sampling). |
| Table 3 | "20 random draws" but no variance reported. | Add ± std or say it is below 1 pt. |
| §6 Prevalence | "At 35% (not shown)". | Add the row to the table or the appendix. |
| §6 Prevalence, last sentence | The hybrid "shared half" cannot be disabled, but it has 0–20% recall on unseen framings, so the floor protects only seen framings. | Say so. Otherwise the sentence oversells the hybrid. |
| §3 l.75 | "This shared, held-out-quantile threshold is the distribution-free calibration of Kim and Lee." Kim & Lee use conformal calibration *plus* LEACE register erasure, at ≤5% FPR. | Write: "corresponds to the calibration step of \citet{kim2026linearly}, without their register erasure." |
| Conclusion | "at near-constant false-positive cost": mean FPR triples (0.5 → 1.6–1.8%). | "at a modest false-positive cost (mean FPR 1.6% vs 0.5%)". |

### 1.4 Title

The title is accurate and memorable but runs to three lines. The section 4 heading repeats the main title verbatim, which is fine once but could read as a gimmick. Optional shorter version: *The Gate Knows, the Threshold Doesn't: Label-Free Recalibration of Activation Gates Under Unseen Jailbreak Framings*. Keep "Label-Free" only if the abstract qualifies it (R1).

---

## 2. Style

### 2.1 AI-sounding or formulaic prose

There are no em-dashes, which is good. The main tells are aphoristic contrasts and short "punchline" sentences:

- "The information has not disappeared." / "The failure is a calibration shift, not a representation failure." (abstract). Two punchlines in a row; keep at most one.
- "The information is present; the calibration is wrong." (§4).
- "Attackers do not oblige." (intro).
- "Safety training teaches a language model to refuse harmful requests, and jailbreaks teach it to forget." (intro opener). Cute, but "forget" is inaccurate, since jailbreaks bypass rather than erase. Suggested replacement: "Safety-trained language models refuse harmful requests when asked plainly, yet often comply when the same request is wrapped in a persona, a story, a retrieved document, or an encoding."
- "the collapse is not where one would expect" → "the collapse is in the threshold, not the representation", or simply state the finding.
- "an honest account of the regimes in which it breaks" → "a characterisation of where it fails" ("honest" is self-praise).
- "the exception that proves the rule" → see §1.3.
- "The gains are not free." → "The gains have costs."
- "are solving the wrong problem when the shift is in location rather than in separability" (Discussion). Fine once, but together with the conclusion's "not because … but because" and the abstract's "not a representation failure", the same not-X-but-Y contrast appears at least five times. Keep it in the title and one place in the body.
- "the most principled representational fix we could build" → "a direct representational fix".
- "exactly what a framing held out from training probes" → "what a held-out framing tests".

### 2.2 Passive vagueness or unclear agents

- "are calibrated on the jailbreak framings seen during development" (abstract): say who calibrates. "Their thresholds are set on…"
- "Clusters with fewer than 30 prompts fall back to validation statistics": which statistics (seen-framing median and MAD)?
- "Planned experiments on a 7B model could not be run because GPU capacity was unavailable": remove (R5). Reviewers read it as an excuse.

### 2.3 Undefined or late-defined acronyms

| Term | Issue |
|---|---|
| AUROC / ROC | Used in the abstract and intro before being expanded ("area under the ROC curve" first appears in intro ¶4). ROC is never expanded. |
| ARETE | Never expanded. Either expand it or call it "our geometric gate". The table label "ARETE (v1 geometry)" leaks internal versioning; drop "v1". |
| GCG | Not expanded (Greedy Coordinate Gradient). |
| MLP, PCA, SVD, MAD | MAD is expanded in the intro; the others are standard but expand them on first use for ACL. |
| TPR / FPR / ASR | Defined in §3 Metrics but used before (abstract, intro) in long form. That is acceptable; keep the long form until §3. |
| AIM, DAN | Persona template names; give a one-phrase gloss or cite (shen2024dan). |
| LOFO | Defined in §3; fine. |

### 2.4 Inconsistent terminology

| Concept | Variants found | Recommendation |
|---|---|---|
| Framing | "framing" (85×), "framing family", "wrapper" (8×), "wrapped", "frame" | Use **framing** for the attack family and **wrapped prompt** for an instance. Replace "wrapper" in §4 and Appendix ("wrapper displacements", "each seen wrapper", "held-out wrappers") with "framing". Define both terms once in §3. |
| Gate | "gate", "detector", "probe", "condition vector", "decision rule" | **Gate** = the full decision (score + threshold). **Probe** = the score function. Use "detector" only for prior work. |
| Recall | "recall" (abstract, text), "TPR" (tables, Metrics), "detects X%" | Pick one. Since Metrics defines TPR, use "TPR (recall)" once and then **recall** in prose; make the table headers match, or vice versa. |
| Plain request | "plain request" (text) vs "Direct" (all table headers) | Rename the column to "Plain". |
| Readouts | "mean of the user's tokens", "user-token mean probe", "user mean", "Probe, user tokens (mean)" | Use one name, e.g. **user-mean probe**, everywhere including tables. Likewise **last-token probe** (Table 1 "Probe, last token" vs Table 4 "Linear probe (last)") and "last template token" vs "final-token". |
| Method | "framing-adaptive recalibration", "adaptive recalibration", "adaptive rule", "adaptive gate", "Adaptive recal. (ours)" | Name it once (e.g. **FAR**, framing-adaptive recalibration) and use "FAR" or "FAR-hybrid" in tables and text. |
| Union gate | "multi-view union", "union of the three readouts" | Introduce it in §3 with the other gates, not first in §4. |
| Linear probe | "logistic-regression probe" (§3) vs "linear probe" (§4, Table 4) | Choose one. |

### 2.5 Spelling convention

British spelling is used consistently (standardise, behaviour, characterisation, normalised), apart from "defense" (American) next to "behaviour". ACL accepts either variety but not a mix. Use "defence" throughout, or switch to American spelling.

---

## 3. Citations

### 3.1 Key existence

All 49 cited keys exist in `references.bib`; there are no missing keys. Eleven bib entries are uncited: hughes2025bon, jiang2024wildteaming, blandfort2025redteamprobes, kirch2025features, **farzam2026hiding**, **schwarz2026entanglement**, gao2026alarm, david2026prompted, dubey2024llama3, hendrycks2017baseline, liu2020energy. With natbib, uncited entries do not print, so this is harmless. The bolded two, however, should be cited (§0 item 5):
- farzam2026hiding belongs in Related Work ("Representation-level defenses") and next to ARETE: "Farzam et al. likewise disentangle goal from framing in activations; our negative result for framing-subspace projection suggests the shift is in calibration rather than representation."
- schwarz2026entanglement belongs in Discussion as a contrasting finding (refitting thresholds does not help on topic-matched pairs). Explain the difference: there the information is missing, while here it is retained.
- gao2026alarm and david2026prompted would strengthen "operating-point evaluation" and the LOFO protocol respectively. They are optional but cheap to add.

### 3.2 Claim-support check

| Location | Key | Verdict |
|---|---|---|
| l.33 "linearly readable signal of **harmfulness**" | arditi2024refusal | **Partial.** Arditi is about a *refusal* direction, not harmfulness. Move it to a refusal clause, or keep only zou2023repe and zhao2025harmfulness. |
| l.33 "probe-based detectors flag a prompt before generation" | mckenzie2025highstakes | OK (activation probes for high-stakes interactions), though the domain is not jailbreaks. Consider "probe-based monitors". |
| l.39 "linear jailbreak probes are poorly calibrated off-distribution" | kim2026linearly | Supported. |
| l.39 "distribution shift degrades threshold transfer before it degrades ranking" | luo2026measuring | Supported, nearly verbatim. |
| l.41 "recalibrating subpopulations without labels has antecedents" | hebertjohnson2018multicalibration | **Mismatch.** Multicalibration requires labels. Write "…without labels \citep{ziegler2019unsupervised}, and calibration on identifiable subgroups \citep{hebertjohnson2018multicalibration}…". |
| l.41 median/MAD insensitive to contamination | rousseeuw1993mad, leys2013mad | OK. Rousseeuw & Croux is mainly about Sn/Qn alternatives to MAD but discusses MAD's 50% breakdown point. Acceptable. |
| l.57 "probe- or gradient-based detectors flag prompts before decoding" | mei2026hiddenguard | **Mismatch.** HiddenGuard is a token-level router for in-stream redaction during generation, not a pre-decoding prompt flag. Move it to a separate clause: "or redact during generation". |
| l.57 same | hu2024gradientcuff | **Loose.** Gradient Cuff estimates refusal loss by sampling responses, so it is not strictly "before decoding". Write "detectors that flag prompts from internal signals". |
| l.57 "internal harmfulness scores can even anti-rank successful jailbreaks" | luo2026measuring | Supported. Note the tension: Luo also reports AUROC falling 0.936 → 0.803 under wrapping, yet Discussion l.164 cites Luo for "probes keep their ranking". Drop Luo from the l.164 list, or cite it only for "threshold transfer degrades first". |
| l.57 "final-token probes miss jailbreaks whose evidence sits in earlier user tokens" | doda2026beforelast | **Unverifiable from the notes.** This key does not appear in related_work_bib.md and has no status tag. Verify existence, authorship, and claim. |
| l.63 "re-thresholding often suffices when target labels are available" | azov2026sallie | Notes say the recalibration "appears to use labelled target data". The hedge is appropriate; keep "when target labels are available". |
| l.63 "combine register erasure with distribution-free calibration on labelled benign data" | kim2026linearly | Supported. |
| l.63 self-training under drift | piet2025jailbreaksovertime | Supported. |
| l.63 "test-time adaptation updates normalisation statistics on the fly" | wang2021tent | **Imprecise.** Tent updates normalisation *affine parameters* by entropy minimisation (using test-batch statistics). Write "adapts normalisation layers at test time by entropy minimisation". |
| l.75 "This … threshold is the distribution-free calibration of" | kim2026linearly | **Overstated** (see §1.3). |
| l.111 "harmfulness represented over the user's tokens rather than only at the decision point" | zhao2025harmfulness | **Partial.** Zhao localises harmfulness at the *last instruction token*, not as a mean over user tokens. Write "harmfulness is encoded at the instruction tokens rather than at the final position". |
| l.164 "probes keep their ranking but lose their operating point" | mckenzie, kim, luo, azov | McKenzie, Kim and Azov are OK. Luo is partly contradictory (see above). |
| l.60 refusal-string matching misjudges | souly2024strongreject | Supported (empty jailbreaks). |
| Other uses (attack families, benchmarks, guards, label shift, conformal) | — | Consistent with the notes. |

### 3.3 Cited keys that are unverified ([K] or [U] in the notes, or untagged): spot-check list

**[U] (a detail could not be confirmed):**
- `chao2025pair`: SaTML 2025 venue unconfirmed (earlier NeurIPS'23 R0-FoMo workshop).
- `cui2025orbench`: ICML 2025 venue unconfirmed.
- `sheshadri2024lat`: venue and co-author list partly from memory. The bib already cites it as arXiv.
- `piet2025jailbreaksovertime`: ACM DOI present; venue name (likely AISec 2025) unconfirmed. Entry type is `@article` with an arXiv journal despite having a DOI.
- Partial: `zhang2025jbshield` (full author list from memory) and `bailey2026obfuscated` (tail of author list from memory).

**No tag / not in notes:** `doda2026beforelast` (verify it exists; it supports a substantive claim) and `qwen2024qwen25` (standard; low risk).

**[K] (canonical, not re-fetched):** zou2023gcg, wei2023jailbroken, liu2024autodan, mehrotra2024tap, shen2024dan, anil2024manyshot, yuan2024cipherchat, greshake2023indirect, mazeika2024harmbench, chao2024jailbreakbench, rottger2024xstest, souly2024strongreject, zou2023repe, arditi2024refusal, zou2024circuitbreakers, sharma2025constitutional, inan2023llamaguard, han2024wildguard, zeng2024shieldgemma, saerens2002adjusting, lipton2018bbse, alexandari2020labelshift, wang2021tent, tibshirani2019conformalshift, podkopaev2021labelshiftconformal, hebertjohnson2018multicalibration, rousseeuw1993mad, leys2013mad.

The [K] entries are low risk but should be checked against DBLP or ACL Anthology before camera-ready. The arXiv-only status of zou2023repe and li2023deepinception is worth confirming, as is any 2026 venue for zhao2025harmfulness.

### 3.4 BibTeX hygiene

- **CRITICAL: `note` fields render in the PDF** (see §0). Delete them all or convert them to `%` comments.
- **"and others" in cited entries.** Complete the author lists from the Anthology, arXiv, or the proceedings:
  - `anil2024manyshot` (Anil, Durmus, Panickssery, Sharma, others)
  - `sharma2025constitutional` (Sharma, Tong, Mu, Wei, others)
  - `zeng2024shieldgemma` (Zeng, Liu, Mullins, Peran, others)
  - `qwen2025qwen3guard` (`{Qwen Team} and Zhao, Haiquan and others`). A team author mixed with an individual reads oddly; use either `{Qwen Team}` alone or the full list.
  - `padhi2025graniteguardian` (`Padhi, Inkit and others`). The notes say to fill this from the ACL Anthology.
- **Malformed DOI.** `kim2026linearly` has `doi = {10.18653/2026.findings-emnlp.1648}`. ACL Anthology DOIs have the form `10.18653/v1/2026.findings-emnlp.1648`. Verify it.
- **Abbreviated or inconsistent booktitles:**
  - "Proceedings of ACL (Long Papers)" (mei2026hiddenguard, xie2024gradsafe) and "Proceedings of NAACL-HLT" (rottger2024xstest): give full names, or use Anthology BibTeX.
  - `zhang2025jbshield` booktitle "34th USENIX Security Symposium" should read "Proceedings of the 34th USENIX Security Symposium".
- **Missing fields.** Most `@inproceedings` entries lack `pages` and `publisher` (acceptable for ML venues). ACL entries should carry pages and DOI; Anthology exports supply them. The `eprint` fields are ignored by `plainnat`/`acl_natbib` without `archivePrefix`, so arXiv IDs for conference papers do not print. That is fine, but redundant.
- `anil2024manyshot` has no URL, arXiv ID, or pages. Add the NeurIPS proceedings URL.
- `doda2026beforelast` and `qwen2024qwen25` have no `note`. That is fine once all notes are removed.

---

## 4. ACL formatting

| Item | Status |
|---|---|
| **Template** | Not ACL. Uses `article`, custom margins (0.75 in side, 0.95 in top/bottom vs ACL's 2.5 cm), and numeric `plainnat`. Must switch to `acl.sty` in `[review]` mode, which adds line numbers and the anonymous header, plus `acl_natbib.bst` for author-year citations. `\citet` will then render "Kim and Lee (2026)", so check sentences like "\citet{kim2026linearly} combine…". |
| **Page length** | The current PDF is 10 pages. Main text (Sections 1–8) ends on page 6, Ethics is at the top of page 7, and references and the appendix fill pages 7–10. ACL's text block is about 7% narrower than the current one, so the main text should come to **about 6.5 pages in ACL format**, well within the 8-page long-paper limit. This leaves room for the standalone Limitations section (not counted), a LOFO table, a per-framing FPR column, and the missing shared user-mean baseline row. Re-measure after conversion; four `table*` floats can push pages around. |
| **Limitations** | **Missing as a section.** Required: unnumbered `\section*{Limitations}` after the Conclusion and before Ethics/References. See R5. |
| **Ethics Statement** | Present and adequate. Consider adding the dual-use point (the prevalence-flooding result tells attackers how to disable the adaptive half), licence and terms of use for the public benchmarks, and the Responsible NLP checklist items: compute budget, model sizes, and judge model. |
| **Responsible NLP checklist items in the text** | Not reported: GPU hours/hardware, number of parameters (1.5B is given), variance over the 20 draws/3 seeds, licences of HarmBench/AdvBench/JBB/XSTest/OR-Bench, and the judge prompt. Add a short "Implementation details" appendix. |
| **Anonymity** | "Anonymous submission" is fine and there are no author names, URLs, or acknowledgements. Self-reference "a geometric gate of our own design (ARETE)" is fine for a new contribution. If ARETE was described in a prior non-anonymous preprint, cite it in the third person. The "v1 geometry" label and the GPU-availability sentence hint at a project history; remove both. No code link is present; if one is added, use an anonymised repository. |
| **Hyperref warnings** | `main.log` reports duplicate destinations `table.1`–`table.4` (`table*` combined with `\resizebox{\input}}` across reruns, or the algorithm float). This is harmless, but it means links to tables may jump wrongly. It usually disappears with `acl.sty`; otherwise add `\usepackage[hypertexnames=false]{hyperref}`. |
| **Abstract length** | 296 words; ACL recommends at most 200. See R1. |
| **Algorithm** | Line 1 says "run lines 2–4" from within line 1, and the offline/online split is clear. Fine. Prefer `\State` text in sentence case, without bold labels. |
| **Tables** | `\resizebox` on tables shrinks fonts unevenly. ACL style prefers `\small` or `\footnotesize` with natural width. Table 1 mixes TPR and AUROC in a `\scriptsize` parenthetical, which is hard to read at the ACL width; consider splitting AUROC into its own table or the appendix. |
