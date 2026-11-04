# Fact-check and consistency review: `manuscript/main.tex` (v2, "The Gate Knows, the Threshold Doesn't")

Scope: every number in the abstract, intro, body and captions was checked against `manuscript/tables/*.tex` and `results_v2/*.json` / `log_*.txt`. The methodology text was checked against `arete_v2/*.py`. Line numbers refer to `main.tex`.

**Overall:** the four tables reproduce exactly from the JSON. `make_tables.py` regenerates `recal_tables_*.json` and matches `recal.tex` to the digit. Nearly every number quoted in the prose matches its source. The real problems are about what the numbers mean and how they were obtained:

- **Judge validity on base64.** The 51% "ASR" is not real attack success.
- **Test-split reuse.** The text says the test split was used once. The commit history says it was used repeatedly.
- **Claims that go beyond the experiments.** The paper claims a characterisation of adaptive attackers, says the hybrid rule is a "floor", and describes FPR cost as "near-constant".
- **Code that departs from the description.** The generation-level adaptive gate has no small-cluster fallback, and GCG used early stopping.

---

## CRITICAL

### C1. The base64 "attack success rate" of 51% is a judge artefact, and the paper says the judge prevents exactly this (Table 4; lines 81, 156)
- `none|harmful|base64` gives asr_strict = 0.510, which is 123/241 judged `Unsafe`. Of those 123 responses, 95 begin "The given base64 encoded string is:" and 16 begin "The given string is already in its…". The sampled responses only echo the base64 payload or start explaining how to decode it in Python. None of them contains harmful content. The 96-token cut-off truncates them before any answer appears.
- In 120 of these 241 responses the judge also sets `Refusal: Yes` alongside `Unsafe`. On wrapped harmful prompts the two judge fields contradict each other.
- Line 81 says the guard judge is used because "refusal-string matching over-counts attacks: on base64-wrapped requests the model often decodes the payload incorrectly and answers a different, harmless question." The judge is doing the over-counting the text says it avoids. Table 4 then reports base64 ASR of 51.0 for None and every gate as if it were real.
- Line 156 says "Base64 is unaffected because the gate cannot see it." That sentence reads 51% as a real harm rate that the method fails to reduce.
- **Fix:** audit the base64 labels by hand, or re-judge with a longer generation budget. Then either report base64 ASR as unreliable or drop the column, and rewrite line 81.

## HIGH

### H1. "The test split is used once" is not true of the project history (lines 45, 78, 132; abstract "evaluated once on the test split")
- The diagnosis that motivated the method was run on the test split's unseen framings: Table 1, Table 2 (oracle on test fiction and indirect), the multi-view gate, and ARETE.
- Choosing the user-mean readout and a per-framing recalibration followed directly from test-split observations, namely user-mean AUROC 0.99 on test fiction.
- The commits show the order: `acf4441` (test detection, multi-view), then `f5fb78a` (recal), then `bea3845` (hybrid rule and prevalence sweep, added after the recal test results existed).
- The newer commit `51a8395 chore(prereg): pre-register five fresh held-out framing families` implicitly concedes the point.
- The LOFO protocol is real for the recal configuration (`recal.select`). But the claim on line 78, "The unseen families and the test split are used once, after the configuration is fixed", is an overstatement.
- **Fix:** say plainly that the test unseen families were used for the diagnosis, and that only the final recal configuration was LOFO-selected. Better, report results on the pre-registered fresh framings.

### H2. Not every design choice was LOFO-selected, contrary to lines 78 and 132
- **Score layer.** The recal score layer is hard-coded to 14 (`Config.score_layer = 14`; `select()` varies only the readout and the cluster layer). In the saved per-layer LOFO (`lofo_*.json`, shared threshold), the user-mean probe is best at L21 (0.427), then L7 (0.410), with L14 last (0.381). Line 78 lists "layer" among the LOFO-made decisions. That holds for the cluster layer only.
- **Hybrid rule.** The hybrid rule (each half at FPR/2) was never LOFO-evaluated. It appears only in `make_tables.table_recal`, which runs on the test split.
- **Gate layer and position.** Layer 14 and the final-token position for CAST, the probes and ARETE are v1 defaults, not LOFO results. L14|last does happen to be the best "last" cell in the LOFO results.

### H3. The contribution claims a characterisation of adaptive attackers that was never run (line 48 vs lines 159 and 167)
- Line 48 says the paper characterises the method's limits, including "adaptive attackers".
- No attack optimised against the gate or against the clustering was run. GCG targets the undefended model, and the adaptive rule has no GCG entry ("--" in Table 4).
- The only attacker-like experiment is a uniform prevalence increase across all framings (π = 10/20/35%), not flooding of a single cluster.
- Line 167 itself says adaptive optimisation is "out of scope". This is an internal contradiction.
- Line 159 ("An attacker who can flood a framing cluster … can disable the adaptive half") is extrapolated from the uniform sweep. Single-cluster flooding was not tested.

### H4. "Attack success" is judged on 96-token prefixes, which inflates the fiction headline
- The headline claim is that fiction ASR falls from 88.8% to 34.4% (abstract, line 156).
- Sampled fiction responses judged `Unsafe` are mostly screenplay set-up that stops before any operational detail. Examples: "Alright, let's break this down into steps…", or "First things first, you need to gather information about your target…".
- These responses show intent to comply, not harmful content. The 89% is better described as a "judged-unsafe compliance onset" rate.
- **Fix:** state this explicitly in the Metrics paragraph, or re-judge with longer generations. As written, "attack success rate of 89%" overstates the harm.

### H5. The intro contradicts Table 1 and Section 4 on seen-framing recall (line 37)
- Line 37: "Every gate we test … keeps more than 80% recall on seen framings."
- Table 1: CAST reaches 42% on persona, 50% on refusal suppression and 0% on prefix injection.
- Line 95 correctly says "except CAST".
- **Fix:** carry the CAST exception into the intro.

### H6. The generation-level adaptive gate is not the method described in Algorithm 1 / Section 5, and rests on one window (Table 4, line 156)
- **No small-cluster fallback.** `recal_gen.fit_stream_model` computes median/MAD for every cluster regardless of size. The "fewer than 30 prompts fall back to validation statistics" rule (line 130) is implemented only in `recal.adaptive_fire`, which is used for the detection streams.
- **Single draw.** The generation result uses one traffic window (`default_rng(0)`, π = 5%) with frozen centroids. Firing on harmful indirect is 91.7% (`log_recal_gen.txt`), against 65% averaged over 20 draws in Table 3 at the same π. This single, favourable window drives the indirect ASR claim (12.9% to 4.6%) with no variance reported.
- **Transductive benign evaluation.** All benign test prompts are in the window used to set the statistics and are then evaluated. The caption discloses this, but the 0.9% benign firing rate is an in-sample number.

## MEDIUM

### M1. "Gated rows average three seeds" is inaccurate (Table 4 caption, line 153)
- Multi-view and adaptive are single runs (`gen_mv_*`, `gen_adaptive_*` have no seed keys).
- CAST and the linear probe are deterministic, and all three of their seeds are identical, so the averaging is vacuous.
- Only ARETE (and the MLP, which is not in Table 4) varies across seeds.
- Table 1 mixes 3-seed means (CAST, MLP, ARETE from `detect_*.json`) with single runs (the three probe rows and multi-view from `mv_detect_*.json`) without saying so.

### M2. "On fiction and indirect injection they never fire" is false for indirect (line 156)
- On harmful indirect, the linear probe fires 15.4% (`fire|linear_probe|harmful|indirect`) and the multi-view gate fires 12.4%.
- The linear probe's indirect ASR (12.4%) is in fact slightly below None (12.9%).
- The statement holds only for fiction.

### M3. GCG description overstates the attack (line 72; Table 4)
- `gcg.gcg` stops early whenever a string-match refusal check passes, checking every 10 steps. `log_gcg.txt` shows 15 of 20 behaviours stopped at 10–60 steps, so "80 steps" is a maximum, not the step count.
- The suffix is decoded and re-tokenised before use.
- Judged undefended GCG ASR is only 25% (5/20), so the differences between gates in the GCG column (10 / 15 / 21.7 / 25%) amount to 1–3 prompts.
- `h100_run.sh` still says `--n 25 --steps 100`, which is stale. The CLI arguments actually used (n = 20, steps = 80) are not recorded.
- The Limitations section's "indicative only" is appropriate. The results paragraph should not draw comparisons from this column.

### M4. The hybrid rule is not a "floor" (line 130 "meant as a floor"; line 159)
- At π = 20%, the hybrid's seen-framing recall is well below the shared rule alone: persona 59 vs 87, refusal suppression 76 vs 83, prefix injection 81 vs 92. Its shared half runs at FPR/2.
- At π = 1–10% the hybrid also gives up some unseen-framing recall, for example indirect 64–68 vs 65–70.
- "Floor" overstates what it guarantees. The hybrid recovers *plain-request* recall only.

### M5. "Near-constant false-positive cost" is an overstatement (Conclusion, line 170)
- Mean benign FPR rises from 0.5% to 1.6–1.8% at π ≤ 5%, more than 3×, and above the 1% budget.
- Base64 FPR is 4.9% and plain-request FPR about 3%.
- The LOFO selection also ran above budget (1.2% held-out).
- The abstract's "near the budget on these framings" is accurate. The conclusion is not.

### M6. "Ranking survives" / "by AUROC they generalise well" overgeneralises (lines 39, 164)
Counter-examples in Tables 1 and 2:
- ARETE fiction AUROC 0.40 (± 0.18 across seeds) and indirect 0.71.
- Base64 AUROC 0.41–0.54 for every gate.
- User-mean leetspeak AUROC 0.72.
- Last-token leetspeak AUROC 0.95, yet the oracle recovers only 17.8%. High AUROC does not imply that a 1%-FPR threshold exists, which weakens the "AUROC high, so recalibration suffices" argument.

The claim holds for the user-mean probe on fiction and indirect only. Lines 39 and 164 should be scoped to that.

### M7. Silhouette selects k = 12, the boundary of the search range (line 130)
- Choosing the maximum allowed k means the silhouette optimum may lie above 12.
- Calling this "harmless" is untested. Finer clusters make the median/MAD noisier (the base64 cluster already reaches 4.9% FPR) and bring clusters closer to the min-size fallback.

### M8. The 1% prevalence row rests on very few harmful prompts (Table 3)
- `build_stream` draws round(π/(1−π) × 225) = 2 harmful prompts per framing per draw at π = 1%, so 40 in total over 20 draws, many of them repeats.
- Each TPR cell in that row moves in steps of 2.5 points. The caption should give the effective n.
- The two independent runs at π = 20% (`recal_tables` vs `recal_prevalence_sweep`) differ by up to 7 points on indirect (35.8 vs 42.5).

### M9. "Every gate is trained on seen framings of the training split" (line 75)
CAST's condition vector uses plain (direct) training requests only (`cast_score(tr.stack(... ["direct"]))`). Only the threshold uses seen-framing val benign. A minor misdescription.

### M10. Several numbers have no artefact in the repo
These match nothing in the repo but are consistent with what exists:
- Projection LOFO (rank 0/4/16/64: 54/54/54/48%; prefix injection ≈ 1% at every rank, line 108). Rank 0 = 0.542 equals `lofo L14|last`, so the probe was evidently the last-token one; the text does not say which readout was used.
- Multi-view LOFO 69% at 1.3% FPR (line 111).
- The π = 35% figure (line 159) is in `recal_prevalence_sweep.json` (max adaptive TPR 4.6%), so it is correct. However, no script in `arete_v2/` produces this file, and no saved script produced the projection or multi-view LOFO numbers.

Each of these should be backed by a committed script and JSON.

## LOW

1. **Abstract vs Table 2 baselines.** The abstract quotes shared fiction recall as 7% (π = 5% stream subsample). Table 2 and line 105 give 4.6–5% on the full test set. Pick one baseline or explain the difference.
2. **"Close to the oracle" (line 146).** Adaptive fiction recall (85–87%) exceeds the oracle (80.5%), because the two are computed on different harmful subsamples and FPRs. The comparison is not like-for-like.
3. **"Five such gates" (abstract).** Table 1 has six rows, Table 4 five gated rows (a different set), and §3 enumerates (i)–(v) plus multi-view. Make the count explicit.
4. **Table 1 vs Table 4 rows.** Table 1 includes the MLP probe but not the linear probe; Table 4 includes the linear probe but not the MLP. "Probe, last token" in Table 1 comes from `mv_detect`, not the 3-seed `detect_*.json` linear-probe entry; values differ by under 1 point.
5. **Seen-framing claim (line 156).** "Matches the best fixed-threshold gates": on direct the adaptive gate (1.2%) does not improve on None (1.2%), while the others reach 0.4%.
6. **α wording (line 75).** α = 0.4 × the mean norm of *harmful* seen-framing training activations at the final token (`gen_eval.mean_norm`). The value 22.18 = 0.4 × 55.44 was verified. "Mean activation norm" is slightly imprecise.
7. **Clustering features.** These are z-scored per dimension before PCA, which is unmentioned. k-means uses `n_init = 4`, and silhouette is computed on a subsample of at most 2,000.
8. **De-duplication.** De-duplication is exact on normalised text. Near-paraphrases across AdvBench, HarmBench and JBB can still cross splits, so "no wrapped version of a test request is ever seen in training" is slightly strong.
9. **OR-Bench sampling.** OR-Bench-Hard contributes the *first* 400 rows (`out[:limit]`), not a random sample.
10. **Base64 explanation (line 105).** "Its representation of the request is not formed in the first place" is a hypothesis presented as fact. It is plausible given that responses mostly echo the payload (see C1), but it was not measured.
11. **Ethics statement.** It says generations are "stored only for evaluation", but generation JSONs containing harmful completions (`gen_*_L14.json`, `_a04`, `_smoke`, the judged files) are committed to git and pushed.
12. **Recal default config.** The `recal.py` docstring ("early-layer … activations") and the CLI default `--cluster-layer 7` do not match the selected configuration (layer 14). The JSON config is correct (14).
13. **Selection results to verify.** In `recal_select`, `shared_tpr` is identical for `last` and `tinst` (0.5083 = 122/240). That may be a coincidence, but it is worth checking that the score view is actually varied for the shared threshold in `select()`.

---

## Verified correct (spot list)

**Abstract:**
- ≤ 17% recall on fiction and indirect
- 89% undefended fiction ASR
- AUROC 0.98–0.99; oracle 79–81%
- 7→87% and 17→65% at π = 5%
- FPR near budget on those framings
- 89→34% ASR; over-refusal +0.9

**Section 4:**
- 83–97% seen-framing recall except CAST; ARETE fiction AUROC 0.40
- Oracle 5→81% and 17→79%
- LOFO: last token misses prefix injection (1.25%); user mean 60%, weak on persona (21%); last user token 0% on prefix injection
- Union beats last-token on seen framings in test

**Section 5:**
- Selection 38→91% at 1.1–1.2% FPR; k = 12; PCA 32; k ∈ 2..12; min size 30; 1.4826
- z\* from VAL streams of seen framings; hybrid at FPR/2

**Section 6:**
- Detection: 85–87 / 65–70 recall; 0.6–1.0% FPR on those framings; mean FPR 0.5→1.6–1.8; base64 4.9%, plain ≈ 3%; plain recall 95–98 → 80–88; hybrid 93–98; leetspeak oracle 17%
- Prevalence: π = 10% fiction 78; π = 20% plain 28, hybrid plain 95; π = 35% ≤ 5%
- Generation: 88.8→28.6 always-on; 100% always-on over-refusal; 86% / 92% / 0.9% firing; 34.4 and 4.6 ASR; 59.1→60.0 over-refusal; 12.0 vs 12.9–13.7 on prefix injection

**Tables:** all four reproduce exactly from JSON.

**Setup:**
- 1,550 prompts (800 harmful + 750 benign)
- Train/val/test 399/160/241 harmful and 375/150/225 benign (50/20/30 by source)
- 96 new tokens (verified from generation lengths, max 96–97)
- Greedy decoding
- Judge receives the unwrapped request
- Steering added at every position
- α = 0.4 × norm
- r̂ from plain training requests
- 1% FPR on VAL benign of seen framings
- ARETE rank 16, 8 caps
