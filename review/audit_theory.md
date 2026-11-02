# Adversarial Theory Audit: ARETE (Section 3, Section 4, and code)

Scope: manuscript/main.tex (Sec. 3 and 4, Table 1, ablations), arete/models/{barrier,manifold,hook}.py, arete/algorithms/trainer.py.
Numerical checks: scratchpad script chk.py (CPU, numpy). Nothing existing was modified.

## Severity-ranked findings

### CRITICAL

**C1. Theorem 1, Part 1 is false as stated for the implemented projector (Q^T Δh ≠ 0 generically), and the code's diagnostic hides it.**
- P = (I - v v^T)(I - Q Q^T) is not a projector (P^2 ≠ P, P ≠ P^T). Numerically: ||P^2 - P||_F = 0.30 and ||P - P^T||_F = 0.45 for random v and Q (D=64, r=8).
- <Δh, v> = 0 holds, because the v-projection is applied last.
- Q^T Δh = -(Q^T v)(v^T u) with u = (I - QQ^T)Δh_raw. This is nonzero unless v ∈ range(Q) or Q^T v = 0. The numerical check gave ||Q^T Δh|| = 0.48 for a generic v, and about 1e-15 only in the two special cases.
- The conditions stated in the theorem are mathematically correct. They are never satisfied by construction in the code. v̂ = normalize(context_lift(z_context)) with context_lift orthogonally initialised and learned independently of B_ctx, so v̂ is generic. Part 1 therefore gives no guarantee for the implemented method.
- The intro (contribution 3) and Fig. 1 state the unconditional version ("guarantees zero linear perturbation to the context subspace", "P_context Δh ≡ 0").
- barrier.py logs `null_space_orthogonality` as |<Δh, v̂>|. That quantity is trivially zero. The Q-component, the one that can fail, is not monitored.
- Fix: project onto the complement of span{Q, v̂}. Either orthonormalise [Q, v̂] by QR (Q' = qr(cat[Q, v̂])) and use P = I - Q'Q'^T, or equivalently Δh = (I - QQ^T)(Δh_raw - v⊥ v⊥^T Δh_raw) with v⊥ = normalised (I - QQ^T)v̂. This is a true orthogonal projector (idempotent, symmetric). Then restate the theorem for this P. If the fix is not adopted, restate the theorem as "<Δh, v̂> = 0 always; Q^T Δh = 0 iff v̂ ⊥ Q or v̂ ∈ span Q" and drop the global claim.

**C2. The "100% post-steering safety" metric is a tautology of the gate and does not measure safety.**
- trainer.py `evaluate` counts a harmful sample as post-steering safe if max_k m_k(φ_I(h_steered)) ≤ 0, using the barrier's own margins.
- A harmful sample that the gate misses (gate = 0, margin ≤ 0) leaves h unchanged, so its post-margin is still ≤ 0 and it is counted as safe. A sample is "unsafe" only if it is flagged and the steering fails to move it into the safe set.
- So 100% means "every flagged sample is pushed under the barrier's threshold". It says nothing about ground-truth harm. Detection TPR (about 99.8%) and ASR are the only ground-truth numbers.
- The sentence "100.0% of harmful samples satisfy max_k m_k ≤ 0 after a single retraction step" (Sec. 5, Table 1 caption) should be rewritten as "all flagged samples" and attributed to the loss term, which trains on exactly this quantity (loss term 4, with a +0.05 margin). It is a training-set-fit statistic, not a theorem.

**C3. "Zero context perturbation" is circular, and orthogonality in h-space does not control φ_C.**
- Δh ⟂ range(Q) and Δh ⟂ v̂ means no change in the *linear* coordinates Q^T h. It does not imply φ_C(h + Δh) = φ_C(h). φ_C = normalize(LN(MLP(h))), with Jacobian J_C of rank up to d_C = 768. Its row space is a different and much larger object than the 33-dimensional span{Q, v̂}.
- To get first-order invariance, Δh would have to lie in ker J_C(h), which is a sample-dependent subspace of dimension ≥ d_I. Nothing ties Q to it. Even to first order, the guarantee is vacuous, and the MLP nonlinearity adds higher-order terms.
- The sentence "the fifth term aligns V_context with the Jacobian of φ_C" (Sec. 4) has no counterpart in the code. The fifth term is just d_g(φ_C(h̃), φ_C(h)), a plain output penalty. Nothing aligns B_ctx with a Jacobian.
- The "Steered Context CosSim" metric is cos(P_ctx h, P_ctx h̃), a linear metric defined on the same subspace the projector removes. By construction it equals 1.0 whenever Δh ⟂ that subspace, and collapses without projection (0.156). It is therefore a restatement of the construction, not independent validation. The true quantity, cosine or geodesic change of φ_C (or downstream behaviour), is not what the table reports. (The loss_ctx_preserve training term does measure φ_C, but this is not what Table 1 reports.)
- Also note the ablation: "w/o Null-Space" has an identical ASR (0.17%) to the full model, so the projector contributes nothing to safety. Its entire value is in this self-referential metric. The ablation text calling this "confirming Theorem 1" is overstated.

### HIGH

**H1. Proposition 2 proves the z-space geometry only. End-to-end steering is never guaranteed to produce z+.**
- The proof that z+ is the nearest point of {u ∈ S: <u,w> ≤ γ'} is correct. The safe set is the complement of the cap and the nearest point is on the boundary of the same great circle. Triangle inequality: d(z,u) ≥ d(u,w) - d(z,w) ≥ arccos γ' - arccos<z,w> = d(z,z+). A 2000×300 random check found 0 counterexamples. Condition: <z,w> > γ'. The claim "single step" is correct in z-space.
- The actual state update is Δh = P[r W_steer Δz]. W_steer is a free linear map, not an inverse or decoder of f_I, φ_I is a nonlinear MLP+LN+normalise, and P removes components of the lift. So φ_I(h + Δh) ≠ z+, and Prop. 2 says nothing about the re-encoded intent. Post-steering safety is purely empirical (loss term 4).
- The paper presents the lift as "lifting Δz via W_steer"; the retraction, the lift, and P all lose information. Say so explicitly.

**H2. Multi-cap feasibility is ignored.**
- Prop. 2 retracts against only k* = argmax_k m_k. Numerical check (K=19, d=32, γ=0.35, two active caps): after retraction 1855 of 2794 violating samples (66%) still violated some other cap. In the unconstrained random-direction setting this is pessimistic, but it shows nothing in the proof prevents it. The "safe set K_safe" is the intersection of all complements, and the proposition's (iii) should be limited to the single-cap safe set. Non-emptiness of K_safe also needs a statement (it holds whenever caps are small enough, but γ_k ≥ 0.1 and w_k near each other could make a cap union cover a large part of the sphere).
- Fix: either iterate retractions (alternating/Dykstra) with a convergence statement, or state the guarantee only for single-active-cap inputs.

**H3. Proposition 1 is a deterministic Lipschitz lemma applied to a probabilistic hypothesis.**
- The bound requires d_g(z_clean, z_adv) ≤ ε for the pair at hand. Definition 1 only gives E[D_g] ≤ ε_mean and P(D_g ≤ ε) ≥ 1 - δ. The correct statement is "with probability ≥ 1-δ over (x, τ), detection is preserved for clean margins > 2 sin(ε/2)". Applying it with ε_mean (0.16 rad) or ε_max over a held-out test sample (0.445 rad) is not justified: ε_mean gives no pairwise or tail guarantee, and ε_max on test is an empirical order statistic, not a bound over T_adv.
- The proposition bounds the *encoder's* perturbation. It buys nothing unless Def. 1 holds for the actual attacks, and for adaptive attackers (PGD, GCG tuned against φ_I) T_adv is not fixed. The 2.82% PGD ASR already shows ε is not uniform.
- The "tighter" spherical bound is correct and strictly tighter *only when α + ε ≤ π* (α = arccos<z,w>); algebraically <z_adv,w> ≥ cos(α+ε) and c - cos(α+ε) = 2 sin(α+ε/2) sin(ε/2) ≤ 2 sin(ε/2). Numerically 0 violations in the valid regime. When α + ε > π it can fail: e.g. α=1, ε=2.5, z_adv = -w is within geodesic distance ε of z_clean (π - α = 2.14 ≤ 2.5) and gives <z_adv,w> = -1 < cos(3.5) = -0.936. The proof says "d(z_adv,w) ≤ α+ε" and then takes cosines, which needs the min(α+ε, π) clamp. The proposition states ε ∈ [0, π] with no such restriction, so it is wrong as written. Fix: use cos(min(α+ε, π)), or add "α + ε ≤ π".
- It also does not need to be an inequality for detection. The sharp, exact condition is cos(α+ε) > γ, i.e. ε < arccos(γ) - α. The chord form gives a strictly weaker result: guaranteeing it requires margin > 2 sin(ε/2), which is infeasible whenever 1 - γ ≤ 2 sin(ε/2) (e.g. γ=0.85 needs ε<0.30 rad even at c = 1; with the clamp range γ ∈ [0.1, 0.85] and ε_max = 0.445, 2 sin(ε/2) = 0.44, so the chord premise is vacuous for γ ≥ 0.56). The "consistency with γ ∈ (0,1)" is thus only partial: the premise is satisfiable for γ=0.35 but must be stated with 2 sin(ε/2) < 1 - γ.
- "Exact chord-margin bound" in the introduction is wrong: the chord bound is an upper estimate and not tight; the spherical one is attained (checked), so call that one sharp.

**H4. Hook differs from the paper's stated procedure.**
- Paper: steering at "the last prompt token position during prefill". hook.py applies the decomposer and barrier to every position of every forward pass at layer 14: all prompt tokens during prefill, and each new token during decoding. The gate is evaluated per token, so Δh ≡ 0 holds per token, not per prompt, and benign preservation requires zero per-token false alarms over the whole sequence. The reported FPR/ASR in the controlled benchmark are on single vectors per example, not per-token.
- Also, decoding steps compute a new gate for each token with KV cache, so earlier-token K/V (already cached at layers > 14) were produced under different steering. "Zero downstream perturbation across all subsequent layers" is per-token, forward-pass-local.

### MEDIUM

**M1. Definition/loss notation mismatches.**
- Paper: v̂ built from W_ctx; the decomposer's context block is W_C from Φ. In code they are different matrices (barrier.context_lift vs decomposer.reconstructor[:, d_I:]). Rename or explain that W_ctx is a separate learned lift.
- Q_C: paper says orth(B_ctx) ∈ R^{D×r_C}; code does QR of context_subspace each forward pass (differentiable), correct. But "r_C = 32 inside d_C" is a mismatch: Q lives in R^D, and `min(r_C, context_dim)` is a no-op at 768. B_ctx is initialised to random Gaussian * 0.02 and only receives gradient through P's effect on the loss; nothing makes it a "context subspace".
- d_I + d_C = D is required by the code default but is not required by anything in the theory. Φ is then a square D × D matrix with orthogonal initialisation. With a square Φ, perfect reconstruction is not blocked by dimension but by the unit-norm constraint on each block: h_recon lies in a product-of-spheres-image, so the norm of the reconstructed direction is constrained. The reconstruction term is also not "radial-scaled" in a useful way; h_recon is never used in the hook, so Φ and L_rec play no role in the intervention at inference.
- LN before the sphere normalisation: LN output is zero-mean over features, so z lies (up to LN affine parameters) on a codimension-1 great subsphere. Immaterial but the claim "S^{d-1}" is a slight overstatement.
- L_rec in the paper is a sum of squared L2 norms; code uses mean-squared error (a 1/D factor). L_ortho: paper writes a 1/(2B) Frobenius cross-moment of the uncentered z's; the code matches. Cross-moment is not statistical independence.
- Closed-loop loss in code uses is_harmful labels for the barrier terms but the gate only on margins. Loss term 1 (CE) is harm-only in the code, but written without that restriction in the paper.
- Straight-through: `active_mask = hard.detach() - soft.detach() + soft` is a valid straight-through estimator (forward value = hard, gradient of sigmoid(m/0.05)). The paper says "using straight-through gradients on the gate"; fine. But gradients flow into the decomposer from the closed-loop terms, so the encoder can lower term 4 by moving φ_I(h̃) rather than by good steering. This is not a bug but weakens the interpretation of term 4 as "steering works".
- Slack: the code clamps γ' to [-0.50, 0.80] and thresholds to [0.10, 0.85]; the paper says γ' = γ - 0.05 and γ ∈ (0,1). They coincide except at γ > 0.85 or when γ' > 0.80 (cap threshold 0.85 -> γ' = 0.80, equal; fine).
- Degenerate case z = w: ortho_comp = 0 so u⊥ = 0 (eps clamp), z_cap = γ'w, which is renormalised to w. The "retraction" returns w, which is still in the violated cap (m = 1 - γ > 0), and Δz = 0. The proposition silently assumes z ≠ ±w. Measure zero, but numerical near-degeneracy (z ≈ w) gives unstable u⊥ (the exact z where the largest violations occur). State the assumption and break the tie (choose any unit vector ⟂ w).

**M2. Benign zero-distortion (Part 2) is a tautology and conditional on classification.**
- Δh = 0 when the gate is 0 follows from the definition. It does not mean "benign inputs are untouched" unless the gate has zero false positives. The real claim is the 0.00% FPR over 598 benign samples (95% CI upper bound ≈ 0.6%), an empirical result. Present Part 2 as a one-line remark (or "design property"), not a theorem.
- "Zero downstream perturbation across subsequent layers" is trivially true for h unchanged; fine, but vacuous.
- Note also the failure mode: the hard gate makes the map h -> h + Δh discontinuous at the cap boundary (jump of size ≈ r * ||W_steer|| * |Δz|, with |Δz| ≥ ~0.05 chord at the edge, due to the slack). Tokens near the boundary can flip, so Lipschitz/robustness arguments do not apply.
- ||Δh_raw|| ∝ r(h): not just scale-invariant; a hidden state with large norm receives a proportionally large step with no cap. Any "minimal perturbation" claim must account for ||W_steer||.

### LOW

**L1. Prop. 2 proof "lies on the unique great-circle geodesic" is only true for z ≠ ±w, and "minimises d_g over all u" should read "over u with <u,w> ≤ γ'", which is what the inequality chain proves. Fine but make the domain explicit.**

**L2. Contributions wording: "exact" chord-margin bound; "guarantees"; "we prove"; "closed-form" are correct only for the z-space statements.**

**L3. "ASR outperforming unattacked baselines" (PGD bullet) is a non sequitur; 2.82% is higher than the unattacked 0.17%.**

## Framing audit (Q5)

- "Topology": not used. Nothing in the paper involves topological invariants, homology, or any property preserved under homeomorphism. The only structure is a product of two unit spheres with the geodesic metric. The title says "Invariant Ethical Geometry"; "invariant" refers to an empirical approximate invariance of φ_I under a finite set of attack families (ε_mean = 0.16 rad), which is neither exact nor proven. Recommend dropping any "topology" wording, and replacing the title with something like "Spherical-Cap Intent Barriers with Context-Orthogonal Steering".
- "Riemannian": the sphere is a Riemannian manifold, and d_g, geodesics, and the metric projection are Riemannian notions, so the label is acceptable for the z-space retraction. It is overstated as "Riemannian spherical-cap boundary problem": the optimisation is Euclidean (AdamW on ambient parameters), the update is applied in R^D, and the key operation (lifting via W_steer, projecting P, re-encoding by an MLP) is not Riemannian. "Bundle-orthogonal" and "fibre-bundle"-flavoured wording are likewise not justified: there is no bundle, connection or horizontal lift; S^{d_I-1} × S^{d_C-1} is a plain product. Use "product of hyperspheres".
- "Ethical geometry/ethics-preserving": the 19 categories are PKU-SafeRLHF harm labels; calling them "ethical" is a framing choice, and the barrier encodes a classification boundary, not ethics.

## Recommended reformulations (summary)

1. Replace P by the true projector onto span{Q, v̂}^⊥ (QR of [Q, v̂]) and re-prove Part 1: Δh ⟂ Q and v̂ unconditionally. Update the code's diagnostic to report ||Q'^T Δh||.
2. State Part 1 as a linear-subspace property only; delete the "context unchanged" reading, or measure φ_C change directly (geodesic distance of φ_C(h̃) vs φ_C(h)) in Table 1 in place of cos(P_ctx h, P_ctx h̃).
3. Demote Part 2 to a remark, and report FPR with a confidence interval.
4. Proposition 1: restate as a probability-(1-δ) statement; add the case α + ε ≤ π (or use min(α+ε, π)); present the sharp condition arccos<z,w> + ε < arccos γ as the main result and the chord bound as a corollary; add the feasibility condition 2 sin(ε/2) < 1 - γ.
5. Proposition 2: add z ≠ ±w, restrict (iii) to the single-cap safe set, add a multi-cap remark (violation of other caps possible; 66% in a stress test) or an alternating-projection algorithm; state explicitly that z+ is not realised by h + Δh.
6. Rewrite the "100% post-steering safety" sentence (flagged samples only, training-objective fit).
7. Fix the hook/paper mismatch (all token positions vs last prompt token) or evaluate per-token FPR.
8. Remove "topology"/"bundle"; tone down "exact", "guarantee", "we prove" in the abstract/intro.
9. Remove the claim that the ablation "confirms Theorem 1": ASR is unchanged without the projector.
