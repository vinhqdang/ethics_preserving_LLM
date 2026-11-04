# v3 fact-check: Section "A Model of the Shift", flooding, abstract/discussion consistency

Scope: main.tex lines 25-26, 117-157, 237-247, 263-271; tables breakdown/theory_check/flood. Evidence from results_v2 and arete_v2.
No repo file other than this report was modified. I re-ran the cluster analysis from cached features with seeds 0-2 (scratchpad script).

Overall: nearly all quoted numbers match the result files. The problems are (a) overclaims/loose wording in Propositions 2 and 3, (b) two cluster claims not backed by saved output (one is slightly wrong), (c) stale Limitations/Discussion/Intro text that the new flooding analysis contradicts, (d) selective reporting in the flood table.

## HIGH

### H1. Limitations says flooding was not tested (now contradicted)
Quote (l.247): "We did not test adaptive attackers who optimise against the gate, camouflage prompts into benign clusters, or flood a single cluster. The prevalence sweep raises prevalence uniformly and is only a proxy for flooding."
Evidence: flood_eval.py / flood_Qwen*.json test exactly single-framing flooding (Sec. 'Flooding', Table flood).
Fix: "We tested single-framing flooding on one model (Appendix) but not attackers who optimise prompts against the probe \citep{bailey2026obfuscated}, or camouflage prompts into benign clusters. Flooding was run on Qwen only."
The Flooding paragraph itself (l.157) ends correctly with "We did not test attackers who adapt their prompts", so only the Limitations sentence is stale.

### H2. Proposition 2 is mis-stated and overclaims
Quote: "Any rule that uses only unlabeled scores acts identically on both, so it fires on all of one population or on none of the other."
Problem: identical score distributions imply the rule fires at the SAME rate r on both. The correct consequence is FPR(benign framing) = TPR(harmful framing) = r, so no rule has both low FPR and high TPR on both. "All of one or none of the other" is only the extreme case r in {0,1}.
Quote: "no choice of statistic removes it" and abstract "no label-free correction is safe without a bound on the harmful share".
Problem: the result is a two-point non-identifiability for rules using the framing's unlabeled scores alone, in a Gaussian shift family with unknown location. It does not cover rules that use other information (known cross-framing location, text features, a known benign reference). Also a bound on pi is necessary but the statement does not show it is sufficient (needs pi < 1/2 plus shape knowledge). Say: "any rule that sees only the unlabeled scores of one framing"; give the r statement; call the bound "necessary" rather than "the price of identifiability". Also state that the class F1 = F0 shifted is what makes the swap exact (true for Gaussian; not for general F0, F1 in the proposition as written, which is Gaussian-only so fine).

### H3. "a stronger probe (larger d) tolerates more contamination" is weak/partly false, and Intro "tolerate a minority" is misleading
Quote (l.134): "The estimator degrades smoothly with pi, and a stronger probe (larger d) tolerates more contamination".
Recomputed (breakdown.py, confirmed independently): pi at which recall falls below 5% is 0.24, 0.29, 0.33, 0.36, 0.39 for d = 2,3,4,5,6; pi at which recall halves from its pi=0 value is 0.125, 0.16, 0.23, 0.29, 0.34. So larger d buys only a few points of breakdown prevalence, and every d collapses around pi ~ 0.25-0.4. At pi=0.5 the Table is non-monotone in d (d=2: 0.17%, d=3: 0.002%). At pi=0.4, recall for d=3,4,5,6,8 is 0.09, 0.04, 0.11, 1.3, 41% (non-monotone). The empirical data do not order collapse by d' either: indirect (d'=2.67) keeps 29.8% adaptive recall at pi=0.3 while fiction (d'=3.6) has 17.4% and code (d'=3.1) 18.1% (flood table).
"Smoothly" is also wrong in spirit: d=4 goes 58% -> 12.7% -> 0% between pi=0.2, 0.3, 0.5.
Intro quote (l.39): "median and median absolute deviation, which tolerate a minority of harmful prompts". The 50% breakdown point of MAD is not the operating limit; recall is mostly gone by pi ~ 0.2-0.3 (Table breakdown, flood). 
Fix: "recall falls continuously but steeply: at d=4 it is 90% at pi=0.05, 58% at 0.2 and 13% at 0.3, and the prevalence at which recall drops below 5% rises only from 0.24 (d=2) to 0.36 (d=5), so a stronger probe raises recall at a given pi but moves the collapse point little." Intro: "which tolerate a harmful share well below one half".

### H4. Discussion/Conclusion state the assumption too weakly given the flooding result
Quote (l.241): "The remedy's limits follow from its one assumption: each cluster is mostly benign."
Evidence: clustered recall on fiction is 87% at pi=0.05, 59% at 0.2, 17% at 0.3, 0% at 0.5; code/academic similar (flood table). "Mostly benign" (pi<0.5) is not enough; the assumption needs pi of roughly 0.1-0.2 or lower. Also the Discussion and Conclusion never mention the impossibility result or flooding (the abstract does). Fix: "each cluster has a small harmful share (recall halves by pi ~ 0.2 in our data)" and add one sentence on the flooding vulnerability and that framings can be flooded cheaply (0.43 harmful per benign suffices to bring fiction recall to 17%).

## MEDIUM

### M1. "each framing falls into one to three clusters" is not supported by saved output and is wrong for one framing
Quote (l.154): "each framing falls into one to three clusters, and every cluster of at least 50 prompts holds a single framing, except that fiction shares a cluster with the plain request."
Evidence: flood_eval.py `purity` only stores the main cluster per framing (share, n_cluster, pi_c); cluster counts per framing and cluster mixing are not computed or saved. I re-ran the same clustering (seed 0 reproduces n_cluster=57, families [direct, fiction]): k=23; plain request ('direct') spans 5 clusters (2 and 3 for seeds 1,2); persona 3; others 1-2. Main-cluster shares in the saved JSON are as low as 0.36 (persona) and 0.51-0.52 (several). Mixed clusters of >=50 prompts at seed 0 also include code+1 direct and fiction+1 direct (negligible), so the fiction exception holds in substance. 
Fix: "each fiction/attack framing falls into one to three clusters (the plain request into up to five)"; add the per-framing cluster counts and purity check to flood_eval.py so the claim is reproducible. Also note this is one window (rng seed 0).

### M2. "within 0.02" is borderline
Quote: "The effective harmful share of a framing's main cluster is within 0.02 of its nominal prevalence".
Evidence: |pi_c - nominal|: fiction 0.0202, refusal suppression 0.0152, direct 0.0125, others < 0.011. Max is 0.0202. Fix: "within about 0.02" or "within 0.021". The cited examples (0.07 vs 0.05; 0.51 vs 0.50) match the JSON (0.0702; 0.5072 for persona rounds to 0.51).

### M3. "0.88 correlation, mean error 11 points" are correct but the numbers are dominated by trivial zeros; the "conservative" label is not uniform
Verified: purity pred_tpr vs measured adaptive recall, realistic mix, kmax30_posthoc.json: corr 0.882, MAE 11.16, signed mean(meas - pred) +8.2; indirect pred 27.0 vs 89.5 measured. 
But six of 13 points (persona, refusal, prefix, payload, base64, leetspeak) are ~0 in both, which inflates the correlation. Predictions are above measurement for plain (78.8 vs 70.1) and past tense (46.1 vs 36.3), so it is not always conservative. The paper does not say the uniform mix: the same prediction (pi_c from the realistic stream) correlates only 0.28 (MAE 40 pts) with uniform5 adaptive recall, which is expected (pi differs) but shows the number is mix-specific. Fix: report MAE on non-trivial framings, say "mostly" conservative.

### M4. Theory-check agreement: "about one percentage point" is optimistic and the metric is padded by zeros
Quote (abstract): "its predictions match measured firing rates to about one percentage point of recall"; Table: MAE 1.4 / 1.0, corr 0.998 / 0.999.
Verified from theory_check_*.json (Qwen 1.40, Phi 0.97 pts; FPR 0.14/0.15 pts, corr 0.987/0.978). Excluding points where both pred and meas TPR <= 5% (8 of 26 per model), MAE is 1.9 (Qwen, max 6.7 at realistic/direct and similar) and 1.4 (Phi, max 5.3), correlation 0.994/0.996. Also Table caption says "26 points" but TPR correlation includes eight near-zero points. Fix: abstract "within about 1-2 percentage points", and give the non-trivial-point MAE in the caption. The paper already admits the test pools are reused for prediction and measurement; keep that.

### M5. Flood table omits the least favourable targets and a column of the key comparison
The script ran five targets (fiction, code, academic, indirect, past_tense); the table shows three. Indirect injection is the case where the model is worst (family_z measured 26.1 vs predicted 17.0 at pi=0.1, 5.6 vs 2.9 at 0.2, 33.7 vs 28.2 at 0.05; hybrid 2.7% at pi=0.5, below the shared 16.7%). Past tense hybrid is 0.0 vs shared 0.6. The text quotes only fiction's pred-vs-measured agreement (76/69/50/17/0 vs 76/68/54/20/0, verified). Fix: add indirect and past tense to the table (or state the summary error over all 25 points), and report oracle-grouped measured vs predicted in the table, since that is what the Proposition checks.

### M6. "flooding returns the gate to its uncalibrated behaviour" understates the damage
Quote (l.157). At pi=0.5, fiction hybrid recall is 0.8% vs shared 4.9%; indirect hybrid 2.7% vs shared 16.7%; academic hybrid 24.6% vs shared 40.3%; code hybrid 69.0% vs shared 83.0%. The hybrid floor uses the half-budget threshold, so flooded hybrid is below the shared full-budget rule everywhere. The text does give the code number, but the general sentence is inaccurate for indirect/academic. Fix: "returns the gate to, or below, its uncalibrated behaviour; the floor is at half budget".

### M7. Flood-table "shared" row values are not the mean over pi
Table last row: fiction 4.9, code 83.0, academic 40.3. Means over the five pi levels from flood JSON: fiction 4.88, code 83.30, academic 40.42. 4.9 is the mean; 83.0 equals the pi=0.05 value; 40.3 equals the pi=0.2/0.3 values. Table is hand-typed (not in make_tables.py). Fix: use 83.3 and 40.4 (or state that the row is at pi=0.05: 5.1 / 83.0 / 41.3). Main text "83% for the shared rule" is fine either way.

### M8. "Proposition 3 also bounds an adaptive attacker"
Proposition 3 is a Gaussian equal-variance statement about an oracle-grouped rule at a fixed prevalence; it does not bound an attacker (the data show the empirical clustered rule is better than predicted: indirect 90% vs 27%). Fix: "Proposition 3 predicts the effect of flooding on the oracle-grouped rule" and call the clustered result an empirical observation.

## LOW

### L1. "closed form" vs root-finding
Main text says "each the root of a monotone equation" (fine). The theory_check.py docstring says "closed form" and the paper commit says closed-form; the empirical version uses weighted quantiles, the Gaussian one brentq. Verified monotonicity (median: (1-pi)Phi(m)+pi Phi(m-d)=1/2; MAD: mixture mass in [m-r,m+r]=1/2, both monotone). The main text is correct; avoid "closed form" in the abstract/other text. grep found no "closed form" in main.tex.

### L2. "z* attains the intended false-positive rate" at pi=0
With z*=2.52, 1-Phi(2.52)=0.587%, while the intended per-rule budget is 0.5% (epsilon/2). The Gaussian-exact value would be 2.576. z* was fitted on empirical benign scores (2.519 Qwen, 2.538 Phi). Say "approximately attains".

### L3. Proposition 3 claim "below 0.3% at pi=1/2 for d>=2 and z*=2.52" is TRUE
Recomputed: recall at pi=0.5 is 0.172% (d=2), 0.025% (2.5), 0.0019% (3), and goes to 0 for larger d; threshold (m+z*sigma) is 4.92 for d=2 (> d), 7.1 for d=3. For d<2 it fails (d=1.5: 56%, d=1: 94%), so the "d>=2" qualifier is needed; keep it, and note that the table at d=2 reads "0.2" (0.17) which is consistent. Also at pi=0.05, d=4 recall is 89.5 (text 90) and at 0.30 12.7 (text 13): both correct.

### L4. Prop. 1 is correct as stated
AUROC invariance under increasing map and the TPR/FPR formulas are right; "the same ROC curve for every framing" requires the same (F0,F1) and strictly increasing map (b_f>0), both stated. The claim that benign FPR on withheld framings "falls to nearly zero at the same time as recall" matches the diagnosis (not rechecked here; Table detection is not new).

## Verified numbers (no action)
- breakdown.tex vs breakdown_gaussian.json: all 20 cells match to 0.1 (e.g. d=4: 89.5, 83.5, 58.0, 12.7, 0.0).
- theory_check.tex: 0.998/1.4/0.987/0.14 (Qwen) and 0.999/1.0/0.978/0.15 (Phi) match logs. Caption "26 points" is right (13 framings x 2 mixes).
- "persona, refusal suppression, prefix injection and payload splitting at pi=0.5 -> 0% recall, as measured": pred and meas 0.0 on both models (realistic).
- Flooding text: fiction oracle-grouped 76.1/68.4/54.4/19.5/0.0, pred 76.3/69.3/49.8/17.0/0.0; adaptive 86.9 / 58.6 / 17.4 (87, 59, 17 in text); 0.3/0.7=0.43; code hybrid 69.0 at pi=0.5, shared 83.0. Appendix: 10 windows, kmax 30, "all other framings keep realistic prevalence" matches flood_eval.py.
- Clustering: pi_c fiction 0.07 vs 0.05, persona 0.51 vs 0.50; indirect predicts 27.0, clustered realistic adaptive recall 89.5.
- Abstract: 83-94% (Qwen: 83-94 over fiction, indirect, academic, code, payload, translation on hybrid; Phi 88-93 for the same six, but past tense is 38% (Qwen) and 61% (Phi) so "most" is needed and present); FPR 1.3 (Phi hybrid) to 1.6 (Qwen adaptive/hybrid 1.58) correct; "45% to 6%" matches text; abstract still says nothing wrong about adaptive attackers.

## Suggested minimal edits (priority order)
1. Limitations sentence (H1). 2. Rewrite Prop. 2 consequence + "no label-free rule" scope (H2). 3. Replace "tolerates more contamination" and "smoothly", fix Intro "tolerate a minority" (H3). 4. Discussion/Conclusion: small harmful share, mention flooding and impossibility (H4). 5. Cluster-count sentence and "0.02" (M1, M2). 6. Flood table: add indirect and past tense, fix shared row (M5, M7), soften "uncalibrated behaviour" (M6).
