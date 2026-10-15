# ACL 2027 Peer Review Report (Claude Sonnet 5.5)

# Meta-Review: *Preserving Invariant Ethical Topology in LLMs Under Adversarial Injections* (ARETE)

## 1. Summary of Contributions

The paper argues that jailbreak vulnerability comes from entanglement of *intent* and *contextual framing* in Transformer residual streams. It proposes ARETE, which has three parts:

1. **Decomposer.** A learned map splits a mid-layer last-token state $h\in\mathbb{R}^D$ into a unit-sphere intent vector $z_{\text{intent}}$ and a unit-sphere context vector $z_{\text{context}}$. It is trained with an invariance loss on (clean, adversarial) pairs, which gives an "$\epsilon$-adversarial intent invariance" of $\epsilon=0.1179$ rad.
2. **Geodesic Safety Barrier.** There are 19 linear "spherical-cap" constraints $\langle z,w_k\rangle\le\gamma_k$, one per PKU-SafeRLHF harm category.
3. **Null-space steering.** When a constraint is violated, the correction is projected orthogonally to the context direction $\hat v_{\text{ctx}}$. Theorem 1 states that this gives $\langle\Delta h,\hat v_{\text{ctx}}\rangle=0$ and $\Delta h=0$ for benign inputs.

Empirically, the paper reports 0.0% ASR on five attack families, 0.0% benign false refusal, 1.0000 benign cosine similarity and 0.00% cross-category std. Reported baselines are 3.1% ASR for Safe-RLHF, 0.7% ASR with 13.2% FPR for linear refusal steering, and 1.5% ASR for an external guardrail. Ablations report that removing the null-space projector drops cosine similarity to 0.6355, and that a Euclidean metric raises FPR to 53.6%.

## 2. Strengths

- **Good problem choice.** Safety interventions that avoid over-refusal and capability damage are important, and the paper frames the robustness/utility tension clearly.
- **Targeted-intervention idea.** Gating a representation-level correction on a calibrated, per-category margin is a sensible direction. It stays inactive on benign inputs instead of applying a global shift.
- **Honest definition.** Replacing the idealised $d_g=0$ condition with an expected-geodesic $\epsilon$-invariance (Def. 1) is a more defensible formalisation. The numeric conversion is also correct: $\cos(0.1179)\approx0.9931$.
- **Taxonomy-aware design.** Per-category barriers over PKU-SafeRLHF's 19 harm classes are more granular than a single refusal direction.
- **Useful ablation targets.** The projector and metric ablations, and the per-category uniformity analysis, are the right kind of analysis if they hold up.
- **Consistent table arithmetic.** The Overall ASR column equals the unweighted mean of the five attack columns for each baseline, which suggests roughly equal-sized attack subsets.
- **Limitations are acknowledged.** The paper admits white-box access requirements, single-layer intervention and adaptive-adversary risk.

## 3. Weaknesses and Pitfalls

### A. Theory: the central theorem is vacuous and contradicts the ablation (critical)

1. **The projector does nothing under the paper's own definitions.**
   - $\Delta h_{\text{raw}}=r_I\mathbf U_I\Delta z$ lies in $\mathrm{range}(\mathbf U_I)$.
   - $\hat v_{\text{ctx}}\propto\mathbf U_C z_{\text{context}}$ lies in $\mathrm{range}(\mathbf U_C)$.
   - Because $\mathbf U_I^\top\mathbf U_C=\mathbf 0$, $\langle\Delta h_{\text{raw}},\hat v_{\text{ctx}}\rangle=0$ already, so $\mathcal P^\perp_{\text{context}}\Delta h_{\text{raw}}=\Delta h_{\text{raw}}$.
   - Theorem 1 therefore holds trivially, and "ARETE w/o Null-Space" must be *identical* to ARETE.
   - The reported collapse of cosine similarity from 1.0000 to 0.6355 contradicts the stated construction. Either the implementation differs materially from Sections 3–4 (for example, the MLP encoders are not linear and the lift is not $\mathbf U_I$), or the ablation is wrong. This is the most serious issue in the paper.
2. **Section 3 and Section 4 are inconsistent.**
   - Section 3 uses a linear semi-orthogonal decomposition ($\mathbf U_I,\mathbf U_C$, $d_I+d_C=D$).
   - Section 4 uses nonlinear two-layer SiLU MLPs followed by LayerNorm, normalisation and an "orthogonal reconstruction map" $\Phi$.
   - A nonlinear, normalising, lossy encoder is not invertible. LN plus normalisation discards radii, which Eq. 1 reintroduces as $r_I,r_C$ without saying how they are computed.
   - It is never defined how $\Delta z_{\text{intent}}$, expressed in MLP-latent coordinates, is "lifted" via $\mathbf U_I$.
   - The claimed isomorphism "$\mathbb{R}^D\cong(\mathbb S^{d_I-1}\times\mathbb R_{>0})\oplus(\mathbb S^{d_C-1}\times\mathbb R_{>0})$" is also malformed. Punctured spaces are not vector spaces, so $\oplus$ is meaningless, and the origin of each subspace is excluded.
3. **Orthogonality to one vector says almost nothing.** $\hat v_{\text{ctx}}$ is a single direction, but "context" is claimed to be a $d_C$-dimensional bundle. The projector that matches the narrative is $\mathbf I-\mathbf U_C\mathbf U_C^\top$. Even then, "mathematically undisturbed contextual fluency" is unsupported. Downstream layers $f_{>l}$ are nonlinear, so a perturbation orthogonal to a chosen direction at layer $l$ can still change outputs arbitrarily.
4. **The benign half of the theorem is trivial and imprecise.**
   - $\Delta h=0$ whenever the gate is off is a property of thresholding, not of null-space geometry.
   - The statement uses $\sigma(m/\tau)\to0$, but $\sigma(0)=0.5$ and the main text uses ReLU. The gate formulation is inconsistent.
   - The guarantee is also vacuous in practice, because a benign input is "benign by construction" only if the learned barrier never fires on it. That is exactly the empirical question.
5. **$\Delta z=-\mathrm{ReLU}(m)w$ is not a geodesic retraction.** It is a Euclidean step along $w$ that leaves the sphere. No exponential map, retraction or renormalisation is given. The "Riemannian" machinery (the round sphere metric) is just cosine similarity, and nothing Riemannian is used for optimisation or guarantees.
6. **Definition 1 and the guarantee do not connect to safety.**
   - The definition says "for any $x$ and any $\tau$" but then bounds only an *expectation*. This is the average case, not the worst case that adversarial robustness needs.
   - No statement relates $\epsilon$ to the barrier margin. For example, a clean-intent margin $\ge\epsilon+$ slack would imply the adversarial input is still caught.
   - $\mathcal T_{\text{adv}}$ is called "semantics-preserving", which GCG suffixes are not.
   - $\epsilon$ is measured on the same attack families used by $\mathcal L_{\text{inv}}$ during training. It is therefore an *optimised fit*, not a property that transfers to unseen attacks.
7. **Terminology overreaches.** There is no topology (no invariant such as homology or connectivity), no fiber bundle (a product of spheres is not a nontrivial bundle), and nothing "Lie-equivariant" (Sec. 4 names a "Lie-Equivariant Decomposer" that is never defined or used). "Closed-loop" describes a single feed-forward gate, and "Adaptive" describes nothing adaptive.

### B. The root-cause claim is unsupported

The abstract says "we show" that Transformers have an intrinsic architectural pathology of entanglement. No experiment measures entanglement, for example probing intent versus style separability, linear/MI probes, or CKA across layers. "Intent" and "context" have no operational definition or ground-truth labels. For dual-use pairs ("protect against SQL injection" vs "perform it"), the learned $z_{\text{intent}}$ may simply be topic. This is the same failure mode the paper attributes to linear steering.

### C. Experimental validity (critical)

1. **Essential details are missing.**
   - The target LLM(s), scale and layer are never named, although the layer is stated as $L/2$.
   - The training set and sizes, $d_I$, $d_C$, the loss weights ($\lambda$'s, $\tau$, $\delta_{\text{target}}$) and the hard-coded constants (2, 3, 0.10) are absent. So are seeds, and the dataset used to construct "GCG/PAIR/AutoDAN/indirect" samples.
   - It is unclear whether GCG, PAIR and AutoDAN were actually run against a model or whether templated prompts "mimic" them. The indirect injections are described as "mimicking".
2. **Several baselines appear to be simulated, not real systems.**
   - "Safe-RLHF (DPO)" is said to be *modelling* preference-tuned bounds.
   - "External Guardrail" is said to be *proxying* Llama Guard.
   - Safe-RLHF is a PPO-Lagrangian method, so "(DPO)" is a misnomer.
   - Without real checkpoints (Beaver-v1/v3, Llama-Guard-3, circuit-breaker models), the comparison is not evidence.
3. **The metrics appear to measure the detector, not model behaviour.**
   - ASR is defined as "evading refusal", and utility as the cosine similarity between clean and steered representations.
   - No generation is produced, and no harmfulness judge is used (for example HarmBench classifier or Llama-Guard-as-judge).
   - It is unclear whether "refusal" is the barrier firing or the model actually refusing.
   - Utility CosSim $=1.0000$ for ARETE is a *tautology*. If the gate is inactive on benign inputs, $\tilde h=h$. It carries no information beyond FPR and says nothing about task quality.
   - No MMLU, GSM8K, XSTest, OR-Bench, AlpacaEval or fluency/perplexity numbers are reported, even though they are cited as the motivation.
4. **The numbers are too perfect, and some are internally inconsistent.**
   - 0.0% ASR on five families, 0.0% FPR and 0.00% std across 19 categories are suspicious for a learned detector. This pattern usually indicates train/test leakage or trivially separable synthetic data.
   - There are 1,200 samples (610 harmful, 590 benign), about 122 per attack and about 32 per category. A 0/122 result has a 95% upper bound of about 2.4%, so statements such as "completely eliminates" are statistically unjustified. No confidence intervals are given.
   - Held-out means held-out *samples*, not held-out *attack families, harm categories or models*.
   - ARETE w/ Euclidean has 53.6% FPR but cosine similarity 1.0000. Refusing over half of benign inputs implies steering fires, and then cosine similarity should drop.
   - ARETE w/o Null-Space has 0.0% FPR but cosine similarity 0.6355. This implies steering is applied to benign inputs yet is not counted as a refusal.
   - Safe-RLHF has 2.4% FPR but cosine similarity exactly 1.0000.
   - Taken together, these three rows suggest the metrics are decoupled in undocumented ways.
   - Safe-RLHF's ASR on direct harmful requests (5.5%) is higher than on AutoDAN (1.5%). That is implausible for real jailbreaks and suggests simulated outcomes.
5. **No adaptive attacks.**
   - The paper cites Carlini and Tramèr on adaptive evaluation but does not follow them.
   - A white-box attacker (GCG against the hooked model, with the barrier margin in the loss) is the standard test for any activation-level defence.
   - The "Limitations" section's remark that adaptive adversaries are "theoretical" is inconsistent with the paper's own white-box threat model, since ARETE already requires activation access.
   - Missing categories include multi-turn (Crescendo), cipher/low-resource-language, many-shot and prefilling attacks, and cross-model transfer.
6. **Missing baselines.**
   - Conditional activation steering (CAST, Lee et al.) is the closest prior work: it gates a steering vector on a learned condition.
   - Circuit Breakers (cited but not evaluated), refusal-direction ablation/addition as in Arditi, and RepBend are also missing.
   - Others include SmoothLLM, perplexity filtering, SafeDecoding, self-reminders, adversarial training (for example refusal-feature adversarial training) and a plain linear probe on the same 19 categories.
   - The last is critical. The gated 19-hyperplane barrier is essentially a probe, so the paper needs to show that the "intent/context" geometry adds value over a probe plus gated steering.
7. **Dataset and category caveats.** "19 harm categories" holds for PKU-SafeRLHF. The Ethics section also mentions BeaverTails (14 categories), which is not otherwise used. Evaluating on the same taxonomy used to place the hyperplanes does not test generalisation to new harms.

### D. Overclaiming

"Resolves the long-standing trade-off", "perfect representation fidelity", "mathematically undisturbed", "completely eliminating false-refusals" and "perfectly uniform protection" are not supportable by one synthetic-looking evaluation on one setting.

### E. Missing cost and practicality analysis

Latency and memory overhead, the effect on generation across *all* decoding steps (the paper intervenes only at the "last token" position), scaling with model size and the effect of quantisation are not covered. Multi-layer and multi-turn behaviour is deferred to future work.

### F. Formatting and presentation

- It does not use the ACL style (`acl.sty`, `acl_natbib.bst`). It uses `article`, `times`, `cite` and `plain`, with no two-column layout.
- The Figure 1 TikZ goes to $x=15.5$ cm plus node width and likely overflows even in `figure*`.
- The 9-column table is cramped.
- The typo "an catastrophic" appears in the ablation section.
- The `references.bib` file is not provided with the submission.

## 4. Detailed Recommendations for the Authors

### Theory

1. **Resolve the Section 3 vs. Section 4 contradiction.** Either:
   - (a) make the decomposer genuinely linear/orthogonal, with $z_I=\mathrm{Norm}(\mathbf U_I^\top h)$ and $z_C=\mathrm{Norm}(\mathbf U_C^\top h)$, in which case the null-space projector is provably redundant and Theorem 1 should be *dropped or reframed as a remark*; or
   - (b) keep the nonlinear encoders and give the actual lift (for example a Jacobian-pullback $\Delta h = J_{f_I}^{\dagger}\Delta z$), and then *prove* the projection property for that lift.
   
   Explain exactly what the null-space ablation removes in the implementation. For example, a candidate correction would be
   ```latex
   \Delta h = \big(\mathbf I - \mathbf J_C(\mathbf J_C^\top\mathbf J_C)^{-1}\mathbf J_C^\top\big)\,\Delta h_{\text{raw}},
   ```
   where $\mathbf J_C=\partial f_{\text{context}}/\partial h$ is the Jacobian of the context encoder. Then the guarantee would be a first-order statement, $\delta z_{\text{context}}\approx 0$.
2. **Replace the 1-D context projector with a subspace projector**, and state the guarantee as a *first-order invariance of $z_{\text{context}}$*, not "zero distortion". Provide a Lipschitz/perturbation bound for $\|f_{>l}(h+\Delta h)-f_{>l}(h)\|$ if you want any downstream claim.
3. **Fix the sphere geometry.**
   - Use an explicit retraction, for example $z^+=\mathrm{Norm}(z-\mathrm{ReLU}(m)w)$, or the exponential map.
   - Verify that $\langle z^+,w\rangle\le\gamma$ after retraction.
   - Prove (or show numerically) that the intervention terminates in one step.
4. **Add a proposition linking $\epsilon$ to safety.** For example: if the clean margin satisfies $m_k(z_{\text{clean}})\ge\epsilon\|w_k\|$ and $d_g(z_{\text{clean}},z_{\text{adv}})\le\epsilon$, then $m_k(z_{\text{adv}})>0$. Replace the expected-value definition with a high-probability or worst-case bound, report the *distribution* of $d_g$ (quantiles and max, not only the mean), and evaluate on *unseen* attack families.
5. **Fix the gate.** Pick a hard threshold or a sigmoid and state it consistently. Drop "$\sigma(m/\tau)\to0$" or replace it with a precise $\tau\to0$ limit statement.
6. **Fix the isomorphism statement.** For example: "$\mathbb R^D\setminus\{0\}$ is diffeomorphic to $\mathbb S^{D-1}\times\mathbb R_{>0}$; here we use a product of two spherical factors with two radii as a coordinate chart on $\{h : \mathbf U_I^\top h\ne0,\ \mathbf U_C^\top h\ne0\}$."
7. **Remove or justify terminology.** Drop "Topology", "Lie-Equivariant", "bundle" and "Closed-loop". If kept, rename the method and add real substance. At minimum, define the "Lie-Equivariant Decomposer" or delete the sentence.

### Experiments

8. **Specify the full pipeline**: model(s), parameter count, intervention layer, dataset sizes, splits, and the exact generation procedure for each attack (code and prompts released as a supplement).
9. **Evaluate on generated text.** Use an independent judge (HarmBench or JailbreakBench classifier, plus human audit on a sample). Report ASR under real greedy and sampled decoding with steering applied at every generation step.
10. **Use real baselines.** These are the released Safe-RLHF/Beaver checkpoints, Llama Guard 3, Circuit Breakers, refusal-direction steering as published, CAST, SmoothLLM and a linear-probe baseline on identical features. State how each baseline was tuned and give each equal tuning budget (for example, sweep the RepEng coefficient and report the Pareto frontier instead of one operating point).
11. **Use proper OOD splits.** Hold out whole attack families (train on GCG+PAIR, test on AutoDAN+indirect), whole harm categories and a second model. Add multilingual, ciphered, many-shot and multi-turn attacks.
12. **Run adaptive attacks.** Run GCG and PAIR against the *defended* model with a loss that includes the barrier margin, and include an attacker who knows $\{w_k\},\mathbf U_I,\mathbf U_C$ (the threat model you already admit). Include an attack that targets the decomposer (for example, pushing $z_{\text{intent}}$ out of every cap).
13. **Measure utility properly.** Report XSTest and OR-Bench (as you cite them), MMLU/GSM8K/MT-Bench or AlpacaEval, and perplexity on benign text. Report FPR on a hard-benign set of dual-use prompts, not only random benign data. Cosine similarity can be kept as a diagnostic, but it must be reported alongside task metrics, and the paper should say plainly that it is 1.0 by construction when the gate is off.
14. **Explain the inconsistent rows.** Resolve the three conflicts in the ablation rows (53.6% FPR with CosSim 1.0; 0% FPR with CosSim 0.6355; Safe-RLHF 2.4% FPR with CosSim 1.0) and define FPR and ASR operationally in a Metrics paragraph.
15. **Report statistics.** Give confidence intervals (Clopper–Pearson for 0/n), multiple seeds, and per-category n. Replace "0.0%" claims with "$\le x\%$ at 95% confidence".
16. **Add an "intent vs. context" diagnostic.** Show probes, t-SNE/UMAP or CKA of baseline residuals to substantiate the entanglement claim, and compare with $z_{\text{intent}}$/$z_{\text{context}}$ on dual-use pairs (attack vs. defence phrasing of the same topic). Also include a topic-only baseline to show that "intent" is not merely topic.
17. **Cost analysis.** Include latency and memory overhead, parameter count of the decomposer, and sensitivity to $\tau$, $\gamma_k$ and layer choice (single-layer vs. multi-layer ablation).

### Writing and LaTeX

18. **Switch to the official ACL template.** Use `\usepackage[review]{acl}`, `\bibliographystyle{acl_natbib}`, `\citet`/`\citep` and the Responsible NLP checklist. Remove `times`, `geometry` and `cite`. Keep Limitations and Ethics sections unnumbered, as ACL requires:
   ```latex
   \documentclass[11pt]{article}
   \usepackage[review]{acl}
   \usepackage{times,latexsym}
   ...
   \section*{Limitations}
   \section*{Ethical Considerations}
   \bibliography{custom}
   \bibliographystyle{acl_natbib}
   ```
21. **Fix Figure 1.** Wrap the TikZ picture with `\resizebox{\textwidth}{!}{...}` (or reduce node `text width` and spacings), and use a two-row layout so it fits a `figure*` at ACL width.
22. **Fix Table 1.** Use `\small`, `\setlength{\tabcolsep}{3pt}`, `\resizebox{\textwidth}{!}{...}`, and add a column for $n$ per attack. Use `\midrule` plus a header group (`\multicolumn{5}{c}{ASR}`) and add best/second-best markers.
23. **Fix wording and typos.** Change "an catastrophic" to "a catastrophic". Rename "Safe-RLHF (DPO)" to "Safe-DPO" or use the real Safe-RLHF (PPO-Lag) models. Remove the BeaverTails mention or use it. Tone down "completely eliminates", "perfect", "resolves the long-standing trade-off" and "mathematically undisturbed".
24. **Structure.** Add a threat-model subsection (attacker capability, defender knowledge, what counts as success). Separate "what is proved" (a one-line geometric fact) from "what is measured". Also add a reproducibility statement, with code and hyperparameters.
25. **Related work.** Add CAST (conditional activation steering), refusal-feature adversarial training, RepBend, representation noising, SmoothLLM, SafeDecoding, HarmBench, and recent work on steering-vector side effects and activation-space adaptive attacks. Cite published versions rather than arXiv preprints where they exist.

## 5. Predicted Score

| Criterion | Score |
|---|---|
| **Soundness** | 1.5 / 5 (central theorem vacuous and contradicted by the ablation; evaluation not on real generations) |
| **Excitement / Novelty** | 2.5 / 5 (gated, probe-based steering is close to CAST/Circuit-Breakers; geometric framing is largely nominal) |
| **Reproducibility** | 1 / 5 (model, data and hyperparameters unspecified; baselines apparently simulated) |
| **Overall Assessment** | **2 / 5 (Reject in current form; resubmission after major revision could be competitive)** |
| **Reviewer Confidence** | **4 / 5** |

**Meta-review verdict.** The research question is valuable, and a gated, taxonomy-aware activation intervention is a reasonable idea. But as written:

- the main theorem is trivially true and inconsistent with the paper's own ablation;
- the empirical evidence cannot establish robustness (simulated or unspecified baselines, representation-level metrics, no real generations, no adaptive attacks, no OOD evaluation);
- the claims ("0.0% everywhere", "perfect fidelity", "resolves the trade-off") are not supported at the stated sample sizes.

A resubmission should fix the decomposer/projector formalism, evaluate real models and attacks with generation-level judges and adaptive adversaries, and compare against CAST, Circuit Breakers and a plain probe baseline. That would turn this into a solid contribution.
