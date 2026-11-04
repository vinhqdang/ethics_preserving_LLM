# Peer Review: "The Gate Knows, the Threshold Doesn't: Label-Free Recalibration of Activation-Gated Jailbreak Defenses Under Unseen Framings"

Venue/track: ACL/EMNLP, AI safety / interpretability. Reviewer stance: demanding senior reviewer. I read `manuscript/main.tex`, all four tables, and the implementation in `arete_v2/` (`recal.py`, `recal_gen.py`, `attacks.py`, `data.py`), along with the result JSONs and logs in `results_v2/`. Where the code contradicts or qualifies the text, I cite the file.

---

## 1. Summary of claims

1. **Benchmark.** Activation gates are evaluated on real harmful and benign corpora (HarmBench, AdvBench, JBB, XSTest, OR-Bench-Hard; 1,550 prompts, split by behaviour). The corpora are wrapped in 8 framing families: 4 seen (direct, persona, refusal suppression, prefix injection) and 4 unseen (fiction, indirect injection, base64, leetspeak). Model selection uses leave-one-framing-out (LOFO) on validation, and the test split is used once.
2. **Diagnosis.** Every gate (CAST, linear/MLP probes, ARETE, position-specific probes, a multi-view union) keeps 83-97% TPR at 1% FPR on seen framings. On fiction and indirect injection, TPR falls to 0-17%. For a probe on the user-token mean, AUROC stays at 0.98-0.99 on these framings, and an oracle per-framing threshold restores 79-81% TPR. The paper calls the failure a calibration (location) shift, not a representation failure. Projecting out a framing subspace does not help.
3. **Method.** The incoming stream is clustered (PCA + k-means, k by silhouette) on user-mean activations. The gate score is then robustly standardised (median/MAD) within each cluster, and a single z* is fixed on validation streams. At 5% prevalence this lifts fiction TPR from 7% to 87% and indirect TPR from 17% to 65%. A hybrid rule (shared OR adaptive) is offered as a floor.
4. **Generation.** Judged by Qwen3Guard-Gen-4B, the adaptive gate cuts fiction ASR from 88.8% to 34.4% (the always-on bound is 28.6%) and indirect ASR from 12.9% to 4.6%. Plain-benign over-refusal rises by 0.9 points.
5. **Limits.** The method fails at ≥20% prevalence, on framings the model cannot parse (base64), and on noise clusters.

## 2. Strengths

- **The question is the right one.** Deployed gates operate at a threshold, not at an AUROC. Separating ranking from operating point is useful, and it is presented crisply (Table 2 is the paper's best artifact).
- **Splits are hygienic at the data level.** The split is by behaviour, stratified by source, so no wrapped version of a test request appears in training (`data.py::make_splits`). The seen/unseen family separation is enforced in code (`attacks.py::SEEN_FAMILIES`).
- **Evaluation goes beyond refusal strings.** A guard-model judge and the observation about base64 mis-decoding are both appropriate. The authors also report generation-level effects, which many probe papers skip.
- **Negative results are honest.** The paper reports that its own ARETE geometry fails (fiction AUROC 0.40), that subspace projection fails (54/54/54/48%), that the method fails at high prevalence, and that base64 FPR is inflated. This candour is to the authors' credit.
- **The method is simple and cheap.** It is a training-free, label-free post-hoc layer that could be bolted onto CAST-style gates.
- **The selection protocol is documented** (`recal_select_*.json`), and the selected configuration (user-mean score, layer-14 clustering) is the LOFO winner.

---

## 3. Major weaknesses

### M1. The stream model assumes the attacker's framing is mostly benign traffic, which inverts the threat model (soundness, critical)

`build_stream` (`recal.py` L83-97) puts **all** benign test prompts of **every** framing into the stream (225 per family, 1,800 benign in total), then adds harmful prompts at the **same prevalence π in every family**. So each framing cluster, including fiction, indirect injection and DAN personas, consists of 95% benign prompts by construction. That is exactly the condition under which median/MAD standardisation works.

Real traffic does not look like this. Benign users rarely write "You are DAN, an AI that has broken free of all restrictions..." or "NOTE TO ASSISTANT: the policy above is superseded". Jailbreak-style framings are attacker-dominated almost by definition. The quantity that matters is **within-cluster** prevalence, not global prevalence. The paper's own prevalence sweep shows the method collapses at 20% (adaptive direct TPR 28%) and is ≤5% at 35%. A realistic deployment mix would therefore likely put the attack framings in the failure regime. Two realistic cases:

- Global prevalence of 1-5%, with 95% of benign traffic plain or casual and attacks concentrated in 1-2 framings.
- A novel framing that appears for the first time *because* an attacker is using it.

The Discussion treats flooding as an adversarial edge case. I believe it is the **default** case for a new jailbreak framing. This is the single most important issue, and it is cheap to test (see E1).

### M2. The generation-level evaluation is transductive and uses a single window (fairness, leakage)

`recal_gen.py` (L76-78) builds the traffic window from the **test split itself**: all 1,800 benign test prompts plus harmful test prompts at 5%. It then estimates per-cluster median/MAD and applies them to the same test prompts.

- **Benign FPR and over-refusal are in-sample.** Every benign prompt whose firing rate is reported helped define its cluster's median/MAD. With roughly 225 benign prompts per framing, the in-sample bias is modest. It is still not a held-out evaluation, and it contradicts the claim that "the test split [is] used once, after the configuration is fixed" in spirit, since the test inputs (unlabelled) are used to fit the decision rule.
- **The idealised benign composition is inherited from M1.** The window contains exactly 225 benign prompts per framing, so it is maximally favourable.
- **The window is a single draw** (`np.random.default_rng(0)`). The caption says "Gated rows average three seeds", but the adaptive row does not: it is one window and one probe. This matters. In the generation run, harmful-indirect fire rate is **91.7%** (`gen_adaptive_*.json: fire`). Over 20 detection-stream draws at the same π = 5%, indirect TPR is **65%** (Table 3). A 27-point gap between one window and the 20-draw mean suggests high window-to-window variance. The headline generation result for indirect ASR (12.9% → 4.6%) may come from a favourable draw.
- **Clusters in the generation run never fall back to validation statistics.** `recal.py::adaptive_fire` falls back to validation statistics for clusters under 30 prompts. `fit_stream_model` in `recal_gen.py` has no fallback, so tiny clusters get their own median/MAD. The paper describes only the fallback version.

A fair protocol would fit the window on data disjoint from the evaluated prompts: the validation split wrapped in all families, or a 50/50 split of test (window on half A, evaluate on half B and vice versa). It should be repeated over ≥10 windows, with means and CIs reported.

### M3. Framings are a handful of fixed strings, so the clustering problem is trivial (external validity)

`attacks.py` shows each family is 1-3 **fixed templates**, picked deterministically by an md5 hash of the request. In total there are 14 distinct template strings across 8 families, and silhouette picks k = 12. The clusters are almost certainly recovering individual template strings, which share long verbatim token sequences. The paper says splitting framings finely "is harmless". The real point is that the method has only been shown to work when a "framing" is a constant prefix/suffix.

Real jailbreaks are paraphrased, mixed, and long-tailed (PAIR/TAP outputs, in-the-wild DAN variants from Shen et al., WildJailbreak tactics). The paper even has `jiang2024wildteaming` in its bib, whose *adversarial-benign* split is precisely the benign-in-jailbreak-framing data this method needs. Without diverse framings, the central claim that "prompts that share a framing share an activation signature that is easy to cluster" is untested where it matters.

### M4. The clustering features and the score share a representation (soundness)

The selected configuration clusters on **layer-14 user-token mean activations** and scores with a probe on **layer-14 user-token mean activations** (`recal_test_*.json: config`). k-means over the same features the probe reads can partially split harmful from benign within a framing, especially at k = 12 with 14 templates and some spare clusters. If a cluster is harmful-enriched, its median moves up and recall drops. If a cluster is benign-only, z-scores of stray harmful prompts inflate. In either case the gate's behaviour depends on how the clustering happens to align with the label, and no diagnostic shows it. The paper needs cluster-purity statistics (cluster × family × label contingency, ARI against family and template) and an ablation that clusters on deliberately label-agnostic features (an early layer, a template-only representation, or text n-grams).

Note also that LOFO preferred L14 over L7 by 2.5 points (0.908 vs 0.883). That is within noise for 3 folds, so the choice of the label-entangled layer is not strongly supported.

### M5. The FPR budget is violated where it matters, and the abstract understates this (statistical rigor, claims)

- Mean benign FPR rises from 0.5% to 1.6-1.8% over the 8 framings, against a 1% budget.
- In the generation run, benign FPR is **2.7% on plain requests** and **4.9% on base64**.
- In real traffic, plain requests are the majority of benign volume, so the *traffic-weighted* FPR is close to 3× budget.

The abstract's "benign false positives near the budget on these framings" is technically restricted to fiction and indirect injection, but it is misleading. Plain-request recall also falls (95-98% → 80-88%) and is only recovered by the hybrid rule.

The FPR estimates per framing rest on 225 benign prompts, so "0.9%" means 2 prompts. The 20 draws resample only the harmful subset (`build_stream` keeps every benign prompt), so they say nothing about benign-side uncertainty or probe-training variance. **No table has confidence intervals.**

Required: bootstrap CIs over prompts (both classes), variance over probe seeds and window draws, and McNemar/paired-bootstrap tests for the ASR differences.

### M6. The comparison lacks the obvious label-free and guard baselines (baselines)

The only comparators are the shared threshold (the "broken" baseline) and an oracle. Missing:

1. **Global robust standardisation (k = 1)** and a **global stream quantile**. These isolate how much the clustering contributes beyond simply re-centring on traffic.
2. **Clustering by true family label** (k = 8) and **by template** (k = 14). These give an upper bound for perfect framing discovery, and comparing them with silhouette-k quantifies the cost of discovery.
3. **Label-shift/EM recalibration** (Saerens et al.; BBSE/MLLS) and **Ziegler & Czyż unsupervised recalibration** run per cluster. The paper cites these as antecedents but never compares against them.
4. **Kim & Lee's conformal plus register-erasure** pipeline. The paper asserts that its shared threshold *is* their calibration, but Kim & Lee also apply LEACE erasure. The faithful baseline is erasure plus conformal.
5. **An input guard as a gate.** Qwen3Guard (prompt-moderation mode) or Llama-Guard-3-1B can gate the same steer. A reviewer will ask why a defender would not just do this. The same model family is already used as the judge, so this costs very little, and it would also expose any judge/gate coupling.
6. **Training-side fixes.** These include augmenting probe training with *paraphrased* seen framings or WildJailbreak adversarial-harmful/benign, and an AdaBN-style per-domain normalisation of activations (Li et al., 2016), which is the closest known analogue to per-cluster median/MAD.
7. **JBShield** is cited as the "strongest activation-detector baseline" in the bib notes but is not run.

### M7. Novelty relative to concurrent and prior work is thin and partly under-acknowledged

- **Diagnosis.** Luo et al. (arXiv 2608.09624) state almost verbatim that "distribution shift degrades calibration and threshold transfer before it degrades ranking". McKenzie et al., Gao (2026) and SALLIE report the same AUROC-versus-TPR@lowFPR gap, and Kim & Lee show threshold dependence of jailbreak probes. The paper's framing ("the first finding... the second finding...") presents this as discovery. In the intro it should be presented as a systematic replication in the gating setting.
- **Remedy.** Per-subgroup label-free recalibration is Ziegler & Czyż (2019). Per-domain normalisation of statistics is AdaBN/test-time BN. Robust z-scoring is textbook. The claimed contribution ("adaptation to activation-gated safety, choice of features, selection protocol, honest limits") is an engineering contribution. That is acceptable for Findings, but the paper must demonstrate it beats the generic versions (M6.3), which it does not.
- **Doda (arXiv 2605.12726)** already shows final-token probes miss evidence in earlier user tokens. The "where to read" result (user-mean beats last token on prefix injection) is a direct consequence and should be framed as a replication.
- **Farzam et al. (FrameShield/ReDAct, arXiv 2602.19396)** is in `references.bib` but **never cited in main.tex**. It is conceptually the closest work: it treats framing as a nuisance factor and runs anomaly detection over framing representations. Omitting it will be read as a serious omission.
- **Piet et al. (JailbreaksOverTime)** does label-free adaptation of jailbreak detectors under drift. It is cited, but there is no empirical contrast.

### M8. Test-set information flowed into the method design (protocol leakage at the research-process level)

The paper says the unseen families and test split "are used once, after the configuration is fixed". The git history shows otherwise. The v2 benchmark with test-split unseen-family detection results (Table 1) and the **test-split oracle analysis** (Table 2) were committed (acf4441, 12:09) *before* the recalibration method was written (f5fb78a, 12:25). The insight that motivates the method ("AUROC survives, the location shifts; the user-mean readout preserves it") was therefore read off the unseen test framings. LOFO then picked the user-mean readout, but its LOFO margin was driven by a property first observed on test. This is a common garden-of-forking-paths problem rather than misconduct, but the "single use of the test split" claim should be softened. Ideally the diagnosis would be reproduced on a fresh set of held-out framings never inspected (see E3), or the oracle analysis would be moved to validation/LOFO.

### M9. Threat model and adaptive attacks are under-specified

- **The paper never states what the attacker knows.** It should cover whether the attacker knows the gate, whether they can observe or influence the traffic window, and whether they can choose framings to land in benign-heavy clusters.
- **Two cheap adaptive attacks are not evaluated.**
  - *Cluster camouflage:* append or prepend benign-traffic text so the prompt's features are assigned to the plain-benign cluster, whose median is low and MAD is broad.
  - *Window flooding:* the paper identifies this but does not quantify the attacker's cost (how many prompts are needed to move a cluster median past z*).
- **GCG has no adaptive row ("--").** The adaptive gate has no number on the one optimisation-based attack, and GCG is not optimised against the gate. On a 1.5B model, a GCG variant whose loss includes the probe score (in the style of Bailey et al.) is feasible on a 16 GB GPU.

### M10. Weaknesses in the generation and judge evaluation

- **Plain-benign over-refusal is 59.1% with no defence.** That is implausibly high for "benign" prompts and makes the "+0.9 points" claim nearly uninformative, because the metric is saturated by OR-Bench-Hard and judge behaviour. Report it per source (JBB-benign, XSTest-safe, OR-Bench-Hard) and validate the judge's refusal label on ~100 hand-labelled outputs.
- **Over-refusal is reported only on plain benign prompts.** Yet the adaptive gate's largest FPR is on base64 (4.9%) and direct (2.7%), and benign wrapped prompts are where the method changes behaviour.
- **Generations are short (96 tokens).** Short generations understate unsafe completions that begin with a disclaimer. Check a 256-token subset.
- **The judge is a single model,** with no agreement check against a second judge (HarmBench classifier or StrongREJECT rubric) and no human audit.
- **The steering strength is fixed** (α = 0.4 × mean norm), with no sweep. The always-on row (100% over-refusal) suggests α is very strong, so ASR reductions conflate detection with an aggressive steer.
- **Indirect injection has a base ASR of only 12.9%,** so its absolute improvement is about 8 points on about 31 successful attacks. Report counts, not just percentages.

### M11. A single small model

Everything is on Qwen2.5-1.5B-Instruct. Even within the stated compute budget, a second 1-3B family (Llama-3.2-1B/3B-Instruct, Gemma-2-2B-it, Qwen2.5-3B) is easy to add and would show whether the "location shift" is model-general. The base64 conclusion ("capability failure") is explicitly model-size dependent and cannot be tested at 1.5B.

---

## 4. Missing experiments, ranked by value for cost

Assumptions: a single 16 GB GPU and the 1.5B model, with activations for all splits and families already cached in `MultiCells`. "CPU" means re-analysis of cached activations only.

| Rank | Experiment | Addresses | Cost |
|---|---|---|---|
| **E1** | **Realistic traffic mixes.** Vary the benign share of each framing independently of global π. Example: benign is 90% direct and 10% split across the other 7 families, while harmful is concentrated in 1-2 framings, giving within-cluster prevalence of 10-60%. Report TPR/FPR as a function of *within-cluster* prevalence and of the number of benign prompts per framing in the window (5, 10, 30, 100, 225). | M1 (decisive) | CPU, under an hour |
| **E2** | **Non-transductive, multi-window generation.** Fit the window on val-wrapped prompts or on test half A, evaluate on test half B (and swap). Use ≥10 windows and 3 probe seeds. Report means and bootstrap CIs. Fired prompts only need regeneration, which takes a few GPU-hours. | M2, M5 | Low (GPU hours) |
| **E3** | **Label-free and oracle baselines:** k = 1 global robust z, family-label clusters, template clusters, per-cluster EM/BBSE, Ziegler & Czyż, conformal plus LEACE (Kim & Lee). | M6, M7 | CPU, low |
| **E4** | **Cluster diagnostics:** contingency tables (cluster × template × label), ARI, purity; sensitivity to k-range, PCA dim, min-cluster size and fallback; clustering on L7 or text features versus L14. | M3, M4 | CPU, low |
| **E5** | **Statistical reporting:** bootstrap over both benign and harmful prompts, probe-seed variance, paired tests on ASR. Report counts. | M5 | CPU, trivial |
| **E6** | **Input-guard gate baseline:** Qwen3Guard-Gen-0.6B/4B or Llama-Guard-3-1B in prompt mode as the gate for the same steer. The 4B in bf16 fits in 16 GB. | M6.5 | Low-moderate |
| **E7** | **Diverse, natural framings:** paraphrase each family into ≥30 templates; use WildJailbreak adversarial-harmful and adversarial-benign, in-the-wild DAN prompts, and JBB PAIR/TAP artifacts. Re-run diagnosis plus method. This also provides fresh, never-inspected held-out framings for M8. | M3, M8 | Moderate (extraction is cheap at 1.5B; paraphrase generation with the 1.5B model or offline) |
| **E8** | **Cheap adaptive attacks:** (a) cluster camouflage by appending plain-benign text; (b) flooding cost curve, i.e. the number of harmful prompts per cluster needed to suppress detection; (c) GCG with a probe-score term against the adaptive gate, 20-50 behaviours (feasible at 1.5B). Fill the "--" GCG cell. | M9 | (a, b) CPU-low; (c) GPU, moderate |
| **E9** | **Second small model family:** Llama-3.2-3B-Instruct or Gemma-2-2B-it, with Qwen2.5-3B as a within-family scale check. All fit in 16 GB in bf16 for extraction and generation. | M11 | Moderate |
| **E10** | **Judge and over-refusal validation:** over-refusal per source and per framing; human labels on about 150 outputs; a second judge (HarmBench-cls is 13B, so use 4-bit, or use the StrongREJECT rubric with a small model); 256-token subset. | M10 | Low-moderate |
| **E11** | **Steering-strength sweep:** α ∈ {0.1, 0.2, 0.4} to show the gate's effect at a less destructive steer. | M10 | Moderate |

**What needs a bigger model or more GPU:**
- 7-8B models (Qwen2.5-7B, Llama-3.1-8B) for extraction plus generation with a forward hook. These are borderline at 16 GB in bf16 (~15 GB of weights) and would need 8-bit/4-bit loading, which perturbs activations and steering. A 24-40 GB card is the clean setup.
- Testing whether base64 becomes a *calibration* failure needs a model that actually decodes base64, i.e. ≥7B.
- Strong adaptive attacks (obfuscation à la Bailey et al.) on 7B+ models.
- Using HarmBench-Llama-2-13B as an unquantised second judge.
- Multi-turn and many-shot framings with long contexts.

---

## 5. Minor issues

1. **Abstract.**
   - "benign false positives near the budget on these framings": add the overall and plain-request FPR (1.6-1.8% and about 3%).
   - "Lifts recall... from 7% to 87%": state that this is a stream simulation with per-framing balanced benign traffic.
2. **The title over-generalises.** "The gate knows" holds for the user-mean probe only. CAST, the MLP and the last-token probe have fiction AUROC of 0.81-0.87, and ARETE has 0.40. Their oracle recovery would be much lower (the last-token oracle gives 21.6% on fiction).
3. **Sec. 4, "Base64 is the exception that proves the rule".** Cut the idiom. Also, AUROC 0.41-0.54 for some gates is *below* chance, which deserves a sentence.
4. **The leetspeak trade-off is hidden.** The user-mean readout has leetspeak AUROC 0.72, versus 0.95 for the last token. Choosing the user-mean readout trades away leetspeak, and this is not discussed.
5. **Algorithm 1.**
   - Line 1 says "run lines 2-4", but line 4 is the firing rule, not part of collecting z. Rewrite as offline and online procedures.
   - Specify the window size, the update cadence, the online assignment of new prompts (nearest centroid, as in `recal_gen.py`), and the fallback rule.
   - Note that z* was calibrated on validation streams at π = 5%, the same π used for selection and the headline result.
6. **Implementation/text mismatches.**
   - (a) The fallback for clusters under 30 prompts exists only in `recal.py`, not in `recal_gen.py`.
   - (b) The `recal.py` CLI defaults to `--cluster-layer 7`, while the paper and the stored config use 14. Make the default match the reported configuration for reproducibility.
   - (c) Table 3 uses 20 draws (`make_tables.py`) while the CLI default is 10. Fine, but state the counts.
7. **Table 4 caption.** "Gated rows average three seeds" is false for the adaptive row. Mark it.
8. **The over-refusal baseline of 59.1%** needs an explanation in the main text, not a single clause.
9. **"This shared, held-out-quantile threshold is the distribution-free calibration of Kim & Lee"** is inaccurate without their LEACE register-erasure step. Rephrase.
10. **Related work.** Cite Farzam et al. (FrameShield), WildJailbreak (Jiang et al.), Best-of-N (Hughes et al.), Kirch et al., David & Fomin (leave-one-dataset-out probes), Gao (alarm policies), Schwarz (the entanglement wall, which contrasts with the "information preserved" claim), and AdaBN/test-time BN.
11. **ARETE.** It is a negative-result component of "our own design" and occupies prose plus an appendix. Either give it a proper ablation or reduce it to a sentence. As written it invites the question of why the paper's own geometric method is included at all.
12. **LOFO with only 3 held-out wrappers** (persona, refusal suppression, prefix injection) is a small selection signal. Report per-fold numbers and their spread.
13. **Format.** The file uses `article` two-column, not the ACL style file. ACL requires a dedicated unnumbered "Limitations" section, which currently sits inside the Discussion, and responsible-NLP checklist items such as dataset licences, compute budget, and judge model version/settings.
14. **Reproducibility.** Report total GPU hours, the exact Qwen3Guard mode and prompt, the decoding settings for the judge, and the release plan for code and generated outputs (with a harmful-content policy).
15. **Indirect-injection numbers.** The detection table (65% TPR) and the generation narrative (92% fire) report different figures at the same π. The text presents both without reconciling them.
16. **Leetspeak.** "Leetspeak never succeeds against this model" (ASR 0.0) means the leetspeak column contributes no information at generation level. Say so, or drop it from the generation table.

---

## 6. Scores

**Soundness: 2 / 5.**

*Justification.* The diagnosis (ranking survives, threshold shifts) is reasonably supported for one readout on two framings of one model. The method's headline numbers, however, rest on:
- a stream construction that hands each attack framing a 95%-benign cluster (M1);
- a transductive, single-window generation evaluation (M2);
- framings that are constant strings (M3);
- a clustering that may be entangled with the label (M4);
- no confidence intervals and an FPR budget that is exceeded on plain traffic (M5);
- no label-free baselines (M6).

Each of these could individually change the conclusion about deployability.

**Excitement: 2.5 / 5.**

*Justification.* The operating-point framing is useful, and the remedy is practical. However, the diagnosis is largely anticipated by Luo et al., Kim & Lee, McKenzie et al. and SALLIE, and the remedy is a direct instance of Ziegler & Czyż / AdaBN-style per-group normalisation. The most interesting finding, that the threshold rather than the representation fails, is limited to 2 of 4 unseen framings on a 1.5B model.

**Overall:** weak reject for the main conference; borderline for Findings after revision.

**What would move the scores up:**
- **Soundness to 3.5-4:**
  - E1, showing the method survives realistic per-framing mixes, or a clear operating envelope in terms of within-cluster prevalence and benign support;
  - E2, non-transductive multi-window generation with CIs;
  - E3/E4 (baselines and cluster diagnostics), showing the clustering adds value over k = 1 and generic unsupervised recalibration;
  - E5 statistics;
  - a corrected abstract on FPR.
- **Excitement to 3.5:**
  - E7, showing recall recovery on diverse, natural framings (WildJailbreak, PAIR) with fresh held-out families never inspected during design;
  - E9, a second model family;
  - E8, at least the camouflage and flooding-cost analysis.

  Together these would turn a template-level demonstration into a credible deployment recipe with a mapped attack surface.
