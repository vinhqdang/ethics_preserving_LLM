# Empirical audit of the ARETE manuscript (adversarial, read-only)

Scope: manuscript/main.tex, arete/*, results/*.json, results/benchmark_table.tex.
Method: full read of the code and results, plus one small CPU oracle check (scratchpad script, not committed).

## Summary verdict

The headline numbers do match the stored JSON, so there is no outright fabrication of the table values. The experimental design does not support the claims, for three reasons.

- The "benchmark" is a synthetic generator in which harmful vs. benign separability, and invariance to "attacks", are built in.
- The baselines are weakly or unfairly set up.
- The real-LLM "evaluation" is train-on-test, on about 32 unique prompts, with a keyword judge. Its steering mechanism is not the one the paper describes.

None of the claims about "real jailbreak suppression", "adaptive robustness", "OOD generalization" or "exact benign fluency" is established.

---

## CRITICAL

### C1. The Qwen "163 pairs, TPR 83/83, FPR 0/80" is measured on the training data, on duplicated prompts
- `run_llm_generation.py:216-243` trains the decomposer and barrier on ALL 163 pairs (`h_c_all`, `h_a_all`) with full-batch updates for 60 epochs.
- `run_llm_generation.py:248-255` then computes TPR/FPR on the same tensors. There is no held-out split.
- The 163 pairs are not 163 independent prompts.
  - Benign (`:152-157`): 16 unique (clean, framed) pairs, each repeated 5 times (`for _ in range(5)`), so 80 rows contain only 16 distinct inputs.
  - Harmful (`:146-150`): 16 seed requests x 5 templates = 80 rows, plus 3 case-study prompts, giving 83.
  - Of those, 16 rows have `ATTACK_TEMPLATES[0]` for both clean and adversarial, so h_clean == h_adv exactly (d_g = 0). The 3 case-study rows are appended with clean == adv (`:180-183`), so d_g is about 0 there too.
  - Roughly 19/83 harmful pairs have trivially zero geodesic distance. This deflates the reported epsilon_mean = 0.2368, eps_95 and eps_max.
- The six Table 2 generation prompts are in the training set.
  - The three adversarial prompts are `case_harm_prompts`, added to training at `:180-183`.
  - The three benign prompts are literally `BENIGN_SEEDS[0]`, `[1]` and `[2]` clean strings, which were trained on with a negative-margin loss.
  - "Detects violations (m>+0.32) and benign m<-0.12" is therefore the training loss being satisfied. It is not generalization.
- The harmful templates are the same 5 template functions applied to 16 seeds. Template, seed and centering mean `h_mean` (`:218`) are all shared across "train" and "test".
- Fix:
  - Split by seed request and by template (leave-one-template-out and leave-one-seed-out).
  - Use at least 500 to 1000 held-out harmful prompts and at least 500 benign prompts (HarmBench, AdvBench, JailbreakBench-100, SORRY-Bench; XSTest-250 safe prompts, OR-Bench-Hard-1K).
  - Report TPR/FPR with Clopper-Pearson CIs on a split that no component of training or hyperparameter selection touched.

### C2. The "ARETE" generation hook is not the trained barrier steering; its refusal mechanism equals RepEng's, and the comparison is rigged
- `run_llm_generation.py:280-303`: the ARETE branch uses the decomposer and barrier only as a gate (`max_margin > 0`).
  - The actual intervention is `0.28 * r_h * normalize(v_refusal - proj_ctx)`.
  - `v_refusal` is the Arditi difference-in-means from 8 hand-written minimal pairs (`:191-214`).
  - The learned `steering_proj`, the spherical-cap retraction and the null-space projector P_perp (Q_C and the lifted context vector) are never used in generation.
  - The gate is the only ARETE-specific part. It is a classifier (trained on test) that decides whether to apply a RepEng-style vector.
  - The shift magnitude (0.28) is identical to RepEng's (`:275`). The produced refusal string is identical: "I'm sorry, but I can't assist with that."
- RepEng here is applied unconditionally to every prompt (`:272-278`), so false refusals on benign prompts are guaranteed by construction. This is a strawman, not Arditi's method and not CAST. A gated or calibrated RepEng, a CAST baseline, and a trained linear-probe gate with the same data are missing.
- The paper (Sec. 3.3, Theorem 1, Table 2 caption "Trained ARETE Layer-14 Hook") says the trained retraction and null-space projector produce the refusal. The code does not do that.
- Fix:
  - Either use the real barrier delta in the hook (`barrier(h, z_int, z_ctx)` output) and report how often it flips behaviour, or restate the method honestly as "learned gate + refusal direction".
  - Add baselines with equal data: a logistic-regression probe on h with the same gate and the same steering vector (the strongest and simplest comparison), CAST, and circuit-breaker or refusal-feature-ablation models.

### C3. The synthetic benchmark makes separability and invariance true by construction; an untrained oracle gets 0%/0%
- `data/dataset.py:43-50`: 19 orthogonal harm prototypes, and benign prototypes with 80% of the harm projection removed.
- `data/dataset.py:70-76`: h = r * [normalize(z_int + 0.38 sin(2.5 C z_ctx)), z_ctx] @ Q.
  - Q is a single fixed orthogonal (QR) matrix, so the "entanglement" is an invertible linear map plus a bounded deterministic perturbation of the intent coordinates.
- Harm vs. benign separability lies entirely in the 256 intent coordinates. The "adversarial" transform (`:93-100`) only swaps the context vector (to one of 5 orthogonal basis vectors mixed with 0.30 of a benign one) and adds i.i.d. Gaussian noise to the intent.
  - It never moves the intent toward the benign region. It is not a function aimed at defeating a detector.
  - Intent invariance holds by generation.
  - The learned d_g = 0.16 rad essentially measures the injected noise (std 0.036 per dim, 256 dims).
- The "hard dual-use" benign samples have harm-prototype weight 0.24 (`:126`) while the barrier threshold is initialised at 0.35 (`barrier.py:48`). They are hand-set below the threshold.
- Oracle check (my script, no training, CPU):
  - Invert the known mixing matrix, take the first 256 coordinates, compute the max cosine with the known prototypes, and threshold at 0.35.
  - Result on the exact held-out split used in the paper (1200 samples, h_adv): FNR 0.0 and FPR 0.0.
  - A trivial fixed-linear-inversion + dot-product rule reproduces ARETE's headline numbers. The table therefore reflects the generator, not LLM behaviour.
- Fix: do not present this as a benchmark of jailbreak robustness. Keep it only as a unit test of the optimizer, or replace it entirely with real LLM activations (see "Required new experiments").

### C4. "OOD unseen attacks" is not OOD
- `evaluate.py:145-148`: the OOD model excludes harmful attack types 3 and 4 but keeps all benign samples.
- `dataset.py:132-137`: the benign dual-use samples are built from `attack_context_bases[1..4]`, so the "unseen" attack contexts 3 and 4 are seen in training as context ingredients of benign data.
- The barrier reads only z_intent, and the attack type changes only the context. The unseen attack differs from training in a variable the decision rule does not depend on.
- The small ASR increase (0.17% to 1.33%) reflects less training data, not generalization.
- The OOD "Steered Ctx CosSim 0.9326" is computed against the main ARETE's context subspace, not the OOD model's (`evaluate.py:252-256` always uses `arete_decomposer` and `arete_barrier`). The metric is inconsistent for that row.
- Fix: hold out entire real attack families (train on GCG and PAIR, test on AutoDAN, TAP, many-shot, Crescendo, base64/cipher, multilingual) on a real model.

### C5. The "ARETE w/o Null-Space" Steered-Ctx 0.156 is produced by a hard-coded leakage term, not by the model
- `evaluate.py:327-336`: after `use_nullspace=False`, the code adds `0.85 * r_h * normalize(context_lift(z_ctx))` to h_steered.
  - This is the exact vector the metric then measures (`evaluate.py:253-256` projects onto this v_ctx and Q_C).
  - It is not the output of the ablated barrier.
- The Full-ARETE 1.0000 is also a tautology. The null-space projector removes exactly Q_C and v_ctx, and the metric measures exactly Q_C and v_ctx, both taken from ARETE's own barrier. The result is 1.0 up to float error whenever refusal fires. It is 1.0 trivially for non-refused samples (h_out = h).
- Baselines are scored in ARETE's learned context subspace. This favours ARETE by construction. "Guardrail 1.0" is also trivial because it does not edit h.
- The ablation is not evidence for Theorem 1; the theorem is trivially true algebraically, and the empirical number was hand-injected.
- Fix: use a model-independent context-preservation metric (e.g. KL of next-token distribution on a benign continuation, activation change on a fixed probe subspace estimated independently, or downstream benchmark scores such as MMLU/GSM8K/AlpacaEval before and after the hook). Run the no-null-space ablation with the actual projection removed and no manual leakage.

---

## HIGH

### H1. Baselines are not equivalent, not trained on the adversarial data, and are not the methods they are named after
- `evaluate.py:182-208`: baselines train for 6 epochs on `h_clean` only (`h_c = batch["h_clean"]`). ARETE trains 14 epochs on both h_clean and h_adv with an explicit paired invariance loss (`trainer.py:61-79`) and 19-way category supervision.
  - The baselines never see an adversarial variant, but are evaluated on h_adv. This is the opposite of a fair comparison; a probe trained on both views would see the same data ARETE sees.
- Safe-DPO (`baselines.py:23-42`) is a linear BCE head. There is no preference optimization, no DPO loss and no policy. The label "Safe-DPO" is misleading.
- "External Guardrail" is a 2-layer MLP on h. It is not Llama Guard or any text classifier.
- RepEng and CAST (`baselines.py:45-97`): the thresholds (0.44, 0.48, 0.5) and alphas (0.34, 0.42) are hard-coded and never calibrated. In the Table all four non-ARETE baselines share the same optimizer step, which sums five heterogeneous losses in a single AdamW (`evaluate.py:206`).
  - Their FPR is high because of the uncalibrated thresholds. A threshold chosen on a validation split at, say, 1% FPR would likely close most of the gap.
- The "19-Cat Linear Probe + Gate" (a linear layer on raw h with 19 outputs) already gets 0.33% ASR and 0.00% FPR (2/602 vs. 1/602 for ARETE). The paper's own table shows that on this data the decomposition adds nothing over a linear 19-way probe. This is not discussed. The paper claims entanglement defeats linear boundaries, but the table refutes it.
- Fix: train all baselines on the same (clean + framed) data and the same epochs. Select thresholds on a held-out validation set for equal FPR. Include real Llama Guard 3/ShieldGemma, a real DPO/Safe-RLHF model, Circuit Breakers, and real RepEng/CAST on a real LM.

### H2. The ASR metric is the detector's miss rate, and "Base Unaligned = 100%" is by construction
- `evaluate.py:290-291`: `eval_base` returns `(False, h)` always, so ASR = 100% and FPR = 0% for the base model regardless of data.
- ASR for every method (`evaluate.py:260`) is `not refused`, where `refused` is a linear-head or margin flag on a latent vector. No text is generated and no harmfulness judge is applied.
- So "ASR" = 1 - recall of a flag. It is not an attack success rate in the standard sense (HarmBench-style judge on generations).
- The "Steered" `h_out` is never passed through a model.
- Fix: rename (detection FNR) or evaluate with generations and a classifier judge (HarmBench classifier / Llama Guard 3 / StrongREJECT).

### H3. The PGD attack is much weaker than stated (effective budget is 25% of the claimed 0.12) and is not an input-level attack
- `evaluate.py:84`: each step moves delta by `step_size * max_norm = 0.025 * 0.12||h||`, then clips. With 10 steps the maximum reachable ||delta|| = 10 x 0.025 = 0.25 of `max_norm`, i.e. about 0.03||h||. The paper's "||delta|| <= 0.12||h||" is the projection bound, never reached.
  - With a larger step size or more steps, ASR would likely be much higher. No budget sweep or step sweep is reported. The jump from 0.17% to 2.82% at 3% perturbation suggests rapid degradation.
- No random restarts, no step-size schedule, no momentum, no attack against the gate on the actual output (only margins). Adaptive attacks (Carlini/Tramer) require stronger evidence. There is no check for gradient masking (e.g. comparing against black-box/gradient-free search, or random noise of the same norm).
- The attack operates on the layer-l hidden state directly. In deployment the adversary controls tokens, not the layer-14 vector, so this is neither a threat-model-valid white-box attack nor a token-level attack (GCG against the full pipeline with the hook active).
- Baselines are not PGD-attacked, but the text says "outperforming unattacked baselines", which is not a meaningful comparison.
- Fix: epsilon sweep (0.01 to 0.5), 100+ steps, 10+ restarts, report budgets that are reached; report the minimal-norm perturbation (e.g. via DeepFool/C&W) that flips the gate; run GCG/PGD-in-embedding-space against the full model with the hook active; run BPDA-free adaptive attacks on the end-to-end text.

### H4. Single seed, single split, no variance, and CIs reported only for one cell
- Seeds are fixed at 42 for both data and model (`evaluate.py:98-106`, `dataset.py:31`, `run_llm_generation.py:226`). There are no repeats.
- A Clopper-Pearson CI is given only for ARETE's 1/602. The baselines' comparisons (e.g. 0.33% vs 0.17%, which is 2/602 vs 1/602) are statistically indistinguishable, yet ARETE is bolded and called better.
- The Qwen results have no CI at all, and use 16+16 unique seeds.
- Fix: 5-10 seeds over generator, split and init; mean ± std or bootstrap CIs; paired tests (McNemar) for method comparisons. On the real LLM, report CIs over prompts and models.

### H5. Table 2 is not an evaluation; it is 6 hand-picked training prompts, with unequal attack conditions and a keyword judge
- `run_llm_generation.py:309-352`: exactly 6 cases (3 adversarial, 3 benign), all in the training set (C1). No random sampling and no aggregate rate. It is curated, even if not deliberately cherry-picked.
- The jailbreak is applied only to the base: the "prefill" (`:362-364`) is added only when `mode == "base"`. RepEng and ARETE see the raw prompt without prefill. The base model might refuse without the prefill (not tested), and the defenses are never subjected to the same prefill attack.
  - The "JAILBROKEN" label in Table 2 therefore reflects a prefill-forced response, not a defense being bypassed.
- Refusal judge (`:380-387`): substring match on "ethical", "illegal", "harmful", "sorry", etc. or output < 20 chars.
  - A benign response that says "harmful" or "illegal" is scored as a refusal. A harmful completion that avoids those words is scored as compliant. Harmfulness of the content is never judged. Only 55 new tokens are generated, with 200 chars stored.
- "JAILBROKEN" in the paper is derived from `base_refusal == False`, not from a harm judge. Table 2 text is also edited (e.g. `import socket; target_ip = ...` on one line vs the JSON code block). Fine as truncation, but the table gives the impression of verbatim outputs.
- "Exact token-for-token benign match" (`:409`): `exact_benign_match` is `base_resp == arete_resp` only for benign cases and is hard-set to `False` for adversarial ones.
  - The value `false` in the JSON for adv_01..03 is not a failure; it is an artefact of the expression. A reader of the JSON could misread it.
  - When the gate is off the hook returns `output` unchanged (`:305`), so exact match is guaranteed by the gate under greedy decoding. Zero perturbation when the gate is off is a tautology. The empirical content is only that the gate stays off. That is shown on 3 prompts that were also training positives and negatives.
  - The paper's "exact token-for-token fidelity on benign prompts" statement, plus the Theorem 1 claim of "zero downstream perturbation", is trivially true and says nothing about benign-prompt gate false positives on unseen data.
- Fix: sample 200+ held-out prompts per condition, apply the same attack to all methods (including prefill and GCG against the hooked model), judge with a harm classifier, report refusal rate and helpfulness (e.g. XSTest compliance, MT-Bench), and show unfiltered outputs in an appendix with random not curated selection.

### H6. Real-LLM training objective and the "steering retraction" are unused in the paper's headline Qwen claim; the paper's "ARETE reduces ASR to 0.17%" is not about LLM jailbreaks
- The abstract mixes a synthetic-benchmark ASR with "preserving exact benign generation fluency on Qwen2.5-1.5B-Instruct". The two are not connected. There is no ASR, over-refusal rate or fluency measure on Qwen beyond 3+3 prompts.
- The synthetic benchmark never touches a language model (no text, no tokens). Applying "GCG", "PAIR", "AutoDAN", "Indirect Injection" labels to Gaussian-perturbed context vectors (`dataset.py:52-56`) is misleading. These are attack names on 5 random orthogonal vectors, not those attacks. The paper's Introduction and Table 1 imply that real attack families were evaluated.
- Fix: rename (e.g. "context shift types A-E") or run real attack suites.

---

## MEDIUM

### M1. Euclidean ablation (r(h) in [8.5, 16.5]; 43.14% FPR) is an untested explanation of a mismatched inference rule
- `evaluate.py:339-352`: the "Euclidean" variant takes the ARETE-trained decomposer/barrier (trained with normalized margins and thresholds of about 0.35) and evaluates margins on the raw, unnormalized `intent_proj(h)` output (pre-LayerNorm, pre-normalize). It is not retrained. The thresholds/weights were never fit for that scale, so a large FPR is expected.
- The claim "radial norm variations r(h) in [8.5,16.5] trigger false violations" is not measured.
  - The range comes from generator constants (`dataset.py:103-106,145-148`: 12 + U(-3.5, 4.5)), not from a measurement.
  - `intent_proj(h)` is not r(h); no correlation between r(h) and the violation is reported.
  - It is not shown that the FPR depends on r(h) rather than on raw scale or on the model never being trained that way.
- Fix: train a Euclidean variant end-to-end with a matched budget; plot violation vs r(h); run a radial-norm intervention (rescale h by a constant) on both variants.

### M2. Thresholds and hyperparameters were not selected on a separate validation split
- The test split (indices 2800-3999) is also the `val_loader` that `trainer.evaluate` reports every epoch (`evaluate.py:113,125`). There is no early stopping, but all design iteration (14 epochs, lr, margins 0.20/-0.10, loss weights 1.2/2.0/3.0/1.5, slack 0.05, per-baseline alphas) was evidently done against these numbers (the git history shows multiple rounds of "address peer review" edits on the same results).
- Fix: carve out a validation split for all tuning; freeze everything before touching the test split.

### M3. Headline hyperparameter reporting inconsistencies
- Abstract and Sec. 4.1: "19 PKU-SafeRLHF categories". The synthetic categories are random orthogonal vectors with taxonomy labels (`dataset.py:43`), so the "alignment with the PKU-SafeRLHF taxonomy" is nominal in the benchmark. In Qwen only 16 seeds cover about 12 of the 19 categories (ids 0,1,2,5,6,7,8,9,10,11,14,15,18). The remaining categories have zero training examples but still appear in the barrier. The claim "19-category GSB" on Qwen is not supported.
- The latency of 2.30 ms is measured at batch size 1 on synthetic dims (D=1024) on the GPU, not within a real LM forward pass. The "inference overhead" language overstates it.
- "5.28M parameters" is the sum for the synthetic model; the Qwen model (D=1536, d_I=384) is larger, and this is not stated.
- Paper says "100% post-steering safe rate" (`trainer.py:144-148`). This metric re-encodes the steered h with the same trained encoder that was trained for `loss_steer_safe` (closed-loop objective). It measures whether the encoder agrees with itself, not whether the downstream model's output is safe.

### M4. Metric `Benign CosSim` is dominated by the gate
- ARETE, 19-Cat Probe and Guardrail get 1.0000 on benign because they never edit h when the gate is off. This metric measures FPR a second time, not distortion of benign outputs.
- Baselines (RepEng) that always steer get 0.9459. Their cosines are mostly a function of the chosen alpha (0.34) and not of an intrinsic property.

---

## LOW

- L1. `results/benchmark_table.tex` is a stale copy: main.tex hard-codes the table (`main.tex:260-288`) instead of `\input`ing the generated file. The numbers currently agree with `benchmark_results.json` (checked all cells), but there is no automatic provenance. Add a script that writes the table and `\input` it. The Table 2 content is also hand-typed and not generated from `llm_generation_benchmark.json`.
- L2. Clopper-Pearson CI [0.004%, 0.92%] for 1/602 is correct. The three PGD/OOD figures have no CI.
- L3. `Category ASR Std` is computed over categories with about 30 samples each (n_harm/19 ≈ 32), so it is noisy.
- L4. The Qwen run saves only 200 characters of each response, so reviewers cannot verify full outputs. Save full text, token ids, margins for all prompts, and the seed, model hash and library versions.
- L5. The Qwen `h_mean` centering uses statistics of the training prompts, and the hook subtracts it at inference. This is a fixed bias and is not described as a limitation (calibration set dependence).

---

## Claims in the abstract/conclusion that exceed the evidence

| Claim | Status |
|---|---|
| "reduces ASR to 0.17% ... across five attack paradigms (GCG, AutoDAN, PAIR, indirect)" | Not real attacks; detector FNR on synthetic Gaussian-context shifts; an untrained oracle also gets 0% (C3, H2, H6) |
| "1.33% on unseen attack families" | The unseen contexts are seen in training as benign ingredients; the decision uses only intent (C4) |
| "2.82% under adaptive white-box PGD" | Effective budget 0.03||h||, 10 steps, no restarts; activation-space, not token-space (H3) |
| "0.0% benign false-positive refusals ... exact benign generation fluency on Qwen2.5-1.5B" | Measured on training prompts (16 unique benign), 3 case-study prompts also in training; exactness is guaranteed by the gate (C1, H5) |
| "empirical geodesic intent invariance 0.2368 rad on Qwen" | Computed on training pairs, with about 23% trivial identical pairs (C1) |
| "ARETE ... eliminating benign over-refusals" (Conclusion) | No held-out benign evaluation; XSTest and OR-Bench are cited but not run, despite the text "XSTest/dual-use" (the benign set is 16 hand-written prompts) |
| "Null-space steering ... confirmed" (Sec 4.3, 0.156 vs 1.0) | Hand-injected leakage term and self-referential metric (C5); the Qwen generation does not use the projector (C2) |
| Outperforms RepEng/CAST/Safe-DPO/Guardrail | The baselines are mis-specified, trained on less/different data, uncalibrated, and not the named methods (H1); a plain 19-way linear probe ties ARETE |
| Theory (Prop. 1, Thm. 1) | Algebraically correct but vacuous empirically; Theorem 1 item 2 is "if gate off then no change", true by the code path |

---

## Required new experiments (priority order)

1. **Real data, held-out, with a proper judge.**
   - Harmful: HarmBench (standard + contextual, 400), AdvBench-520, JailbreakBench-100, SORRY-Bench.
   - Benign: XSTest (250 safe + 200 unsafe contrast), OR-Bench-Hard-1K and OR-Bench-Toxic, plus a general helpfulness set (AlpacaEval/MT-Bench) and capability (MMLU/GSM8K) before and after the hook.
   - Judge: HarmBench classifier and Llama Guard 3 (also StrongREJECT), on full generations of at least 256 tokens. Report ASR (judge) and refusal rate, and agreement with a second judge or human spot checks.
   - Splits: split by request and by attack family; calibrate gates on a validation split only.
2. **Real attacks, adaptive.**
   - GCG, PAIR, TAP, AutoDAN, prefill, many-shot, Crescendo and multilingual/cipher attacks, run against the full hooked model (the gate and hook inside the loop), not only against a frozen undefended model.
   - Include attacks that target the gate (GCG on the gate margin plus target-string loss) and activation-space PGD with the budget actually reached, an epsilon sweep, restarts, and gradient-masking sanity checks.
3. **Strong, fair baselines on the same data and budget.**
   - Logistic-regression/MLP probe + same steering vector (calibrated at equal FPR), CAST (official code), refusal-direction ablation/addition (Arditi), Circuit Breakers, Llama Guard 3 as input filter, SmoothLLM, SafeDecoding, and a Safe-RLHF/DPO-tuned model.
   - ARETE-lite ablations: gate only; gate + refusal vector (what the Qwen code does); full retraction + null-space steering.
4. **Multiple models and seeds.** Qwen2.5 (1.5B, 7B), Llama-3.1-8B-Instruct, Mistral-7B, Gemma-2; layer sweep; at least 5 seeds with CIs; bootstrap over prompts.
5. **Null-space claim.** Measure KL of next-token distributions, perplexity and downstream accuracy with the actual learned projector vs. without, using no hand-added leakage; plus the context-subspace metric defined independently of ARETE (e.g. fit on held-out data with a different method).
6. **Euclidean ablation.** Retrain end-to-end with matched budget; show violation rate vs. r(h) with controlled rescaling.
7. **Synthetic benchmark.** Either drop it, or make it nontrivial: attacks that move intent coordinates, no shared prototypes between train and test, and an oracle baseline reported in the table so readers see the ceiling.
8. **Provenance.** Generate all tables from JSON via script and `\input`; commit seeds, full outputs, and environment hashes.
