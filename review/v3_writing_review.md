# Writing review v3 of `manuscript/main.tex` (new Section 4 "A Model of the Shift")

I read `main.tex` in full and checked the compiled `main.pdf` (13 pages in ACL review format). I did not edit the manuscript. Severity: CRITICAL, HIGH, MEDIUM, LOW.

## 0. Priority list

1. **HIGH: the new section is placed well, but three of its paragraphs do not belong in it.** "Checking the predictions", "Clustering error" and "Flooding" use terms the reader has not met yet: the traffic mixes, the "window measurements", k_max, the clustered rule, and the hybrid floor. They are results. Move them to the end of Section 6 as a short subsection, "Testing the model" (section 2 below). That also lets Section 4 end on Proposition 3 and hand over cleanly to the method.
2. **HIGH: the introduction never previews the model.** It appears only as a clause in the contribution bullets. One sentence in finding two and one in the roadmap fix this (section 1.2).
3. **HIGH: Limitations contradicts the new Flooding paragraph.** Limitations says "We did not test ... flood a single cluster. The prevalence sweep ... is only a proxy for flooding." Flooding now tests it. Fix the Limitations sentence (section 3.5).
4. **HIGH: the abstract is about 255 words** and carries three numbers that are hard to trace. A 195-word replacement is in section 3.1.
5. **HIGH: notation clashes.** ε is both the noise term (Section 4) and the false-positive budget (Algorithm 1). d is both the class separation (Props 2 and 3) and the MAD scale d_c (Algorithm 1 line 3). m and σ in Prop 3 are m_c and d_c in the algorithm. See section 4.
6. **MEDIUM: Prop 1 says "the ROC curve is the same for every framing"**, but the paper's own AUROCs range from 0.95 to 1.00 and base64 is near chance. Say the model is a first-order description (section 1.3).
7. **MEDIUM: the page budget is tighter than it looks.** The body ends on page 8, but Table 9 (Phi-3.5) floats to page 9, after the Conclusion and Limitations. A reviewer can read that as a body float beyond the limit. Cuts are proposed in section 5.

---

## 1. Narrative arc

### 1.1 Does the section fit or interrupt?

The order Diagnosis, model, method is right. The diagnosis ends on an empirical puzzle (ranking survives, thresholds do not, and projection does not repair it), and the model explains it before the remedy. The method's first paragraph ("The diagnosis and the model of Section 4 suggest...") then reads naturally.

Three things interrupt the flow:

- **The model opens with notation, not intuition.** The first sentence of the section is a formula. A reader coming from "Where to read" has no sentence telling them why the paper is now doing algebra.
- **The section is long relative to its job.** It runs about 1.3 columns of formalism before the remedy. Half of that is the three verification paragraphs, which look back on a method the reader has not seen.
- **Prop 2 and Prop 3 are about the remedy, not the diagnosis.** They are justified only by the method of Section 5. Placing them before the method is acceptable, but they need one orienting sentence saying so.

Recommended structure (keeps the section, shortens it):

- Section 4 "A Model of the Shift": an orienting paragraph, the model, Props 1 to 3 each with a plain-language lead sentence. End after Prop 3 and its table.
- Section 6 gets a subsection "Testing the model": Checking the predictions, Clustering error, Flooding. These then refer back to Section 4 and forward to nothing.

An alternative, if the page budget needs it, is to dissolve the section. Prop 1 becomes two sentences at the end of Section 3 ("A location-scale reading"), and Props 2 and 3 go into Section 5 right after Algorithm 1 under "What the assumption costs". This saves a heading and an orienting paragraph, but it hides the model, and the author clearly wants it visible. I recommend the first structure.

### 1.2 Does the introduction motivate and preview it?

No. Finding two ends "What moves is the location of both score distributions." The next sentences go straight to concurrent work. Add the model there, and name it in the roadmap.

Replacement for the end of the second-finding paragraph (replaces "We confirm the pattern across gates ... does not repair it."):

> What moves is the location of both score distributions. A location-scale model of this shift (Section 4) accounts for the pattern and has two consequences for any remedy: a label-free correction cannot be safe unless each framing is mostly benign, and the robust statistics we use tolerate a harmful share only up to a limit that the model predicts. Concurrent work reports compatible observations: linear jailbreak probes are poorly calibrated off-distribution \citep{kim2026linearly}, and shift degrades threshold transfer before ranking \citep{luo2026measuring}. We confirm the pattern across gates, readout positions and framings. We also show that projecting out a subspace of framing directions, the most direct representational fix we could build, does not repair it.

Replacement for the third-finding paragraph (removes "report plainly", and puts the model's role in one clause):

> The third finding is a remedy and a map of its limits. Deployed gates see a stream of mostly benign prompts. We cluster the stream by framing, standardise the probe score inside each cluster with the median and median absolute deviation, which tolerate a minority of harmful prompts \citep{rousseeuw1993mad,leys2013mad}, and keep the original threshold as a floor. Unsupervised recalibration of subpopulations has antecedents in general machine learning \citep{ziegler2019unsupervised}; we adapt it to activation-gated safety and test it under a protocol that withholds framings from design. The model of Section 4 predicts where it breaks, and Section 6 checks those predictions against measured firing rates, including under a flooding attacker.

The contribution bullets are fine, but the second bullet is now a list of four things joined by commas. Split it:

> \item evidence that gate failure on withheld framings is mostly a calibration shift; a location-scale model of that shift, with an impossibility result for label-free recalibration and predictions that match measured firing rates; and a negative result for framing-subspace projection;

### 1.3 Are the propositions followable? Intuition before formalism

Currently no proposition has a lead sentence in plain language. Each starts with symbols. Suggested orienting paragraph and lead sentences (replace the first paragraph of Section 4, lines 120):

> Section 3 left two observations side by side: ranking survives a change of framing, and the share of prompts above a fixed threshold does not. The simplest account of both is that a framing shifts and rescales every score it touches without reordering them. We state that account as a model so that its consequences can be checked. It has three, each of which bears on the remedy of Section 5: a fixed threshold must fail, a threshold chosen without labels must assume something about the harmful share, and the robust statistics we use tolerate that share only up to a point.
>
> Let $s=a_f+b_f\,\varepsilon$ be the probe score of a prompt in framing $f$, where $\varepsilon\sim F_y$ depends only on the label $y$, $F_1$ stochastically dominates $F_0$, and the location $a_f$ and scale $b_f>0$ are set by the framing. The framing moves and stretches both classes together.

Prop 1, with a lead sentence and the first-order caveat:

> \paragraph{Proposition 1 (thresholds fail, ranking does not).}
> A framing cannot change how well a probe ranks prompts; it can only change where a fixed threshold falls on the ROC curve. Within a framing, AUROC does not depend on $(a_f,b_f)$, because $\varepsilon\mapsto a_f+b_f\varepsilon$ is increasing, and a threshold $\tau$ gives $\mathrm{TPR}_f=1-F_1((\tau-a_f)/b_f)$ and $\mathrm{FPR}_f=1-F_0((\tau-a_f)/b_f)$. A downward shift of $a_f$ lowers recall and false positives together, which is the pattern in Table~\ref{tab:detection}. The model is a first-order description: it predicts one ROC curve for all framings, whereas measured AUROC ranges from 0.95 to 1.00 and falls to chance when the model cannot parse the framing.

(The caveat matters: "the ROC curve is the same for every framing" is currently an overclaim against Table 2 and the base64 row.)

Prop 2, with a lead sentence and tighter wording (the current "fires on all of one population or on none of the other" is hard to parse):

> \paragraph{Proposition 2 (a label-free rule needs a bound on prevalence).}
> An unlabeled stream cannot tell a benign framing that scores high from a harmful framing that scores low. Take $F_0=\mathcal N(0,1)$ and $F_1=\mathcal N(d,1)$. A framing that is wholly benign at location $a+d$ and one that is wholly harmful at location $a$ both produce $\mathcal N(a+d,1)$. A rule that sees only unlabeled scores treats them alike, so it flags the benign framing or passes the harmful one. Label-free recalibration therefore needs a bound on the harmful share $\pi_f$ of each framing, and no choice of statistic removes that need. Section 5 assumes that each cluster is mostly benign.

Prop 3, lead sentence plus an explicit statement of what is computed (the phrase "each the root of a monotone equation" is jargon that adds nothing; either give the equations or drop it):

> \paragraph{Proposition 3 (the cost of robust statistics).}
> The median and MAD ignore a minority of harmful prompts, but not a large one: as the harmful share grows, harmful scores pull the median up and widen the MAD until the effective threshold clears the harmful mode. Suppose a framing is observed as $(1-\pi)F_0+\pi F_1$ with $F_0=\mathcal N(0,1)$ and $F_1=\mathcal N(d,1)$. Let $m(\pi,d)$ and $\sigma(\pi,d)=1.4826\,\mathrm{MAD}$ be the median and scaled MAD of the mixture, computed numerically. The rule that fires when $(s-m)/\sigma>z^\star$ has [equations as now].

Also fix: the equations use $m+z^\star\sigma$ as a threshold on the *unshifted* scale. State "with $a_f=0$ and $b_f=1$ without loss of generality". Last two sentences of the paragraph are fine; the sentence "The estimator degrades smoothly with π, and a stronger probe (larger d) tolerates more contamination" is the useful one. Keep it, and say which table row gives the d=4 numbers (13% at π=0.30 should be checked against `tables/breakdown.tex`).

Calling Prop 3 a "Proposition" is generous: it is a calculation. "Proposition 1 and 2" have one-line proofs and Prop 3 has none. Either label Prop 3 "Calculation" or add "Proof sketch: direct computation" to each.

---

## 2. Redundancy

| Repeated content | Where | Action |
|---|---|---|
| Benign FPR falls to near zero with recall, so the shift is in location | Diagnosis (Recall collapses) and Prop 1 end | Keep in Prop 1 only, and delete "which already suggests a shift in location: the whole score distribution of a new framing moves below the threshold" from Diagnosis. Or keep it there and shorten Prop 1 to "as in Table 1". |
| The remedy needs "each cluster mostly benign" | Prop 2, Method paragraph 1 ("Two facts help ... which by Proposition 2 is an assumption no label-free method can drop"), Discussion paragraph 2 ("The remedy's limits follow from its one assumption"), Conclusion, Limitations | Keep it in Prop 2 (derivation) and the Discussion (consequence). In Method, cut to "and when a framing is used mostly for benign purposes, the bulk of its scores describes benign prompts (Proposition 2 explains why some such assumption is unavoidable)". |
| Fails when a framing is half harmful, with numbers | Prop 3 text, Table 4 (breakdown), Results "It fails where a framing is mostly harmful", Appendix prevalence | Prop 3 gives theory, Results gives measurements. Delete the second set of "recall below 0.3%" in Prop 3 or point to Table 4 only. |
| Ranking survives, AUROC 0.95 to 1.00, oracle recovers 57 to 81% | Abstract, Intro finding 2, Diagnosis | Intro should give one number, not two. Drop "57--81%" from the introduction (it is in the abstract-like statement in Diagnosis). |
| Projection does not help | Intro, Diagnosis, Discussion last two sentences | Delete "Defenses that remove framing from the representation, as ARETE attempted, address a change in separability. Here the change was in location." from Discussion (see rewrite in 3.4), or move it to Related Work next to FrameShield. |
| Fiction under-segmentation story | Results paragraph 1, Results paragraph 2, Clustering error | After the move, Clustering error and Results paragraph 1 sit next to each other. Merge: Results already says "the loss comes from cluster resolution, not from the statistics" and gives fiction's 76% with true groups. Clustering error can then start from that. |
| Kim and Lee (2026) | Intro, Related Work (twice), Setup, Discussion | Cut the Related Work sentence "linear jailbreak probes conflate benign register with intent" or merge the two calibration mentions into one. |
| Flooding vs prevalence sweep | Flooding paragraph, Appendix Prevalence, Limitations | Settle which is the primary evidence. Flooding is the targeted version, so Limitations should say the prevalence sweep complements it. |

Discussion paragraph 1 is almost entirely a restatement of the abstract and the intro ("Activation gates are usually reported by AUROC ... Deployed gates operate at a threshold, and there they do not. Most of the gap is calibration ..."). Replace it with something forward-looking (see 3.4).

---

## 3. Clarity of new text and rewrites

### 3.1 Abstract (about 255 words; ACL recommends about 200)

The current abstract has three problems besides length. "predictions match measured firing rates to about one percentage point of recall" refers to a consistency check that reuses the test scores (the body says so), so it is stronger than what was shown. "83--94% on two models" should be checked against Table 9, since the text says Phi reaches 61 to 93%. "The method is useful but fragile" is a stock evaluative sentence, and "a post-hoc choice" is better placed next to the number it qualifies.

Replacement (about 195 words):

> Activation-gated defenses read a model's residual stream and steer towards refusal only when a probe fires, with a threshold set on the jailbreak framings known at development time. We test six such gates on real harmful and benign prompts in thirteen framings, nine of them withheld from design. On withheld framings, recall at a 1\% false-positive budget often falls to 0--17\%, although a probe over the user's tokens still ranks harmful above benign prompts (AUROC 0.95--1.00) whenever the model can parse the framing. The failure lies mainly in the threshold. A location-scale model of the shift explains this, shows that no label-free correction is safe without a bound on the harmful share of each framing, and predicts how recalibration degrades as that share grows. We then cluster prompts by framing, standardise the probe score robustly within each cluster, and keep the original threshold as a floor, using no labels from the deployment stream. With enough clusters (a post-hoc choice), recall on most withheld framings reaches [83--94]\% on two models at 1.3--1.6\% mean false positives, and judged attack success on code completion falls from 45\% to 6\% (Qwen2.5-1.5B). The method fails when a framing is mostly harmful, when the steering itself is ineffective (fiction on Phi-3.5-mini), and when the model cannot parse the framing.

The bracketed range needs checking against Tables 5(b) and 9. If the lower bound is 61% on Phi, write "61--94%".

### 3.2 "Checking the predictions" (line 143)

Problems: it names "window measurements of Section 6" before windows are defined; "we build the mixture of the empirical ... compute its median and MAD, and predict" is one long sentence; "isolates the robust-statistics step" is jargon; the paragraph ends with a not-X-and-Y contrast. The claim is also stronger than the evidence: correlations of 0.98 to 0.999 between a prediction built from the test scores and a measurement made on subsamples of the same scores test the arithmetic of the robust step, not the location-scale assumption. The paragraph should say so once, and not twice.

Rewrite (after the move to Section 6):

> \paragraph{Predicted and measured firing rates.}
> Proposition 3 assumes Gaussian scores, so we test it with the empirical ones. For each framing and each traffic mix, we mix the benign and harmful test scores at that mix's prevalence, take the median and MAD of the mixture, and compute the firing rates of the rule that standardises within the true framing. We compare these with the rates measured in the evaluation windows (Table~\ref{tab:theorycheck}). Over the 26 framing--mix points per model, correlation is at least 0.97 and the mean absolute error is 1.0--1.4 points of recall. The model also predicts zero recall at 50\% prevalence for persona, refusal suppression, prefix injection and payload splitting, as measured (0--4\%). Because prediction and measurement use the same test scores, this confirms that contamination of the median and MAD accounts for these failures. It does not show that the rule works on new traffic, which is the question of the rest of this section.

(Note: Results says adaptive recall on seen attack-only framings is 0 to 4%, so "exactly as measured" overstated. Use the 0 to 4% as above.)

### 3.3 "Clustering error" (line 153)

Problems: k_max and "the realistic mix" are undefined at this point; "conservative guide" is ambiguous (conservative in which direction?); "the model predicts 27%" has no stated prevalence; the final sentence is a not-X-and-Y contrast.

Rewrite:

> \paragraph{Clustering error.}
> The oracle grouping is unavailable at deployment, so we ask how much clustering changes the picture. With $k_{\max}{=}30$ on the realistic mix, each framing falls into one to three clusters. Every cluster of 50 or more prompts holds a single framing, except that fiction shares a cluster with the plain request, and the harmful share of a framing's main cluster is within 0.02 of its nominal share (0.07 against 0.05 for fiction, 0.51 against 0.50 for persona). The family-level model therefore follows the clustered rule, with a correlation of 0.88 against measured adaptive recall and a mean error of 11 points, but it errs mostly on the pessimistic side. For indirect injection it predicts 27\% recall at the realistic prevalence and the clustered rule reaches 90\%. A cluster can be tighter than its framing, so its MAD is smaller and its threshold lower. Heterogeneity within a framing, which a single location and scale cannot express, helps the rule here.

Add the prevalence to the 27% claim (5% for indirect injection under the realistic mix). If the 27% is from a different prevalence, say which.

### 3.4 "Flooding" (line 156)

Problems: "(Appendix, Table~\ref{tab:flood})" is not a proper reference; "the harmful prompts it sends are the traffic that goes through" is cryptic; "Proposition 3 also bounds an adaptive attacker" blurs the word *adaptive*, which the paper uses for the recalibration rule ("adaptive rule") and for attackers who adapt to the probe. Rename the attacker "a flooding attacker". Also check one inconsistency: fiction recall of the clustered rule at π=0.05 is given as 87% here, but Table 5(b) and the text report 91% (benign-rich, k_max=30, adaptive). If the flood setup differs (other framings realistic, 10 windows), say so, otherwise reconcile.

Rewrite:

> \paragraph{Flooding.}
> Because the rule standardises within a cluster, an attacker who supplies enough of a cluster's traffic controls its median. We raise the harmful share of one framing from 5\% to 50\% and leave the rest of the stream unchanged (Appendix~\ref{app:extra}, Table~\ref{tab:flood}). For fiction, the oracle-grouped rule falls from 76\% recall to 68, 54, 20 and 0\%, against predictions of 76, 69, 50, 17 and 0\%. The clustered rule falls from 87\% at $\pi=0.05$ to 59\% at $\pi=0.2$ and 17\% at $\pi=0.3$, which takes the attacker about 0.43 harmful prompts for each benign one in the framing; the harmful prompts that go undetected are the ones it sent. The hybrid floor limits the damage only where the shared threshold already works: for code, hybrid recall is 69\% at $\pi=0.5$, against 83\% for the shared rule at full budget. Where the shared threshold fails, as for fiction, flooding returns the gate to its uncalibrated behaviour. We did not test attackers who shape their prompts to the probe \citep{bailey2026obfuscated}, a stronger threat than flooding.

### 3.5 Limitations contradicts Flooding

Replace "We did not test adaptive attackers who optimise against the gate \citep{bailey2026obfuscated}, camouflage prompts into benign clusters, or flood a single cluster. The prevalence sweep raises prevalence uniformly and is only a proxy for flooding." with:

> We did not test attackers who optimise their prompts against the gate \citep{bailey2026obfuscated} or camouflage them into benign clusters. Flooding is tested only for a single framing at a time, with the other framings held at their realistic prevalence.

### 3.6 Discussion (replace paragraph 1)

The current first paragraph repeats the abstract. Suggested replacement that uses the model and tells a practitioner something new:

> Reporting an activation gate by AUROC hides the quantity a deployment depends on, which is where the threshold falls on each framing. Our model makes this concrete: a framing slides the operating point along a fixed ROC curve, so recall at a given false-positive budget can be anywhere on that curve even when AUROC is near 1. Gates should therefore be reported per framing at a fixed budget, as in Table~\ref{tab:detection}. This agrees with observations that probes keep their ranking but lose their operating point off-distribution \citep{mckenzie2025highstakes,kim2026linearly,azov2026sallie}. It differs from settings where the information itself is missing \citep{schwarz2026entanglement}, which is what we see for base64 on a small model. Defenses that remove framing from the representation, as ARETE attempted, target a change in separability; here the change was in location.

Also drop the sentence "The remedy's limits follow from its one assumption: each cluster is mostly benign." In paragraph 2, add a pointer: "The model of Section 4 gives the size of that limit: a probe with separation d = 4 keeps 90% recall at 5% harmful share and loses most of it by 30%."

### 3.7 Other AI-sounding patterns introduced or still present

- Abstract: "The method is useful but fragile." Delete (replaced in 3.1).
- Intro finding 3: "We then report plainly where it works and where it does not." Delete (replaced in 1.2).
- Section 7 end: "The gate is right and the intervention is not." This is a one-line closer and a not-X contrast. Replace with: "The gate fires, but at this strength the steer does not change fiction behaviour on Phi." Next sentence "which is a limit of the intervention and not of the recalibration" becomes "so the limit lies in the intervention."
- Clustering error: "a source of gain here and not of loss" (replaced in 3.3).
- Checking predictions: "shows that contamination explains the failure and not that the method generalises" (replaced in 3.2).
- Results, Generation: "The cost of the hybrid rule falls on prefix injection" is fine. "An always-on steer, by contrast, refuses every plain benign request." fine.
- Diagnosis: "Base64 is the exception." Fine, but "the exception" repeated; keep.
- Prop 2: "it is the price of identifiability, and no choice of statistic removes it." Acceptable but "price of identifiability" is a stock metaphor. Replaced in 1.3.
- "Fiction is the informative failure." Replace with "Fiction on Phi shows a failure of a different kind."
- Section 6 lead sentences "As pre-registered, the method under-segments." and "With enough clusters it works where traffic is mostly benign." read as slogans. Acceptable as paragraph headings in a results section, but the second is a claim that Prop 3 limits; keep.
- Em-dashes: none found. Triads: "gates, readout positions and framings" and "cheap, they leave benign traffic alone, and their operating point ..." are fine.

---

## 4. Notation consistency

| Symbol | Uses | Problem | Fix |
|---|---|---|---|
| ε | noise term in $s=a_f+b_f\varepsilon$ (Sec 4); false-positive budget ε in Algorithm 1 ($\tau_{\epsilon/2}$, $\epsilon/2$ quantile) | Same letter, different meanings, in sections two pages apart | Rename the noise term $u$ or $\xi$ ($s=a_f+b_f\xi$, $\xi\sim F_y$). Keep ε for the budget. |
| d | class separation in Props 2 and 3 and Table 4; $d_c$ = scaled MAD of cluster c in Algorithm 1 line 3 | Same letter for separation and dispersion | In the algorithm use $\sigma_c$, matching Prop 3's $\sigma$. Keep d for separation. |
| m, σ | $m(\pi,d)$, $\sigma(\pi,d)$ in Prop 3; $m_c$, $d_c$ in the algorithm | Prop 3's quantities are population analogues of $m_c,\sigma_c$ | Write $m_c,\sigma_c$ in the algorithm and say in Prop 3 that they are their population versions. |
| z* | Prop 3 and Table 4 (2.52, "fitted on Qwen2.5-1.5B"); Algorithm 1 line 1 sets $z^\star$ at the $(1-\epsilon/2)$ quantile of benign robust z | The caption says "fitted" but the algorithm defines it as a quantile; the normal $(1-0.005)$ quantile is 2.58, not 2.52, so the reader cannot tell what 2.52 is | State "the $(1-\epsilon/2)$ quantile of benign robust $z$ on validation streams; 2.52 for Qwen2.5-1.5B at $\epsilon=1\%$". |
| π, π_f | $\pi_f$ (Prop 2), $\pi$ (Prop 3), "5% prevalence" (Results), "harmful share", "harmful prevalence", "prevalence" | Four names for one quantity | Use "prevalence $\pi$" throughout and drop "harmful share" and "effective harmful share" (or define: share = within a cluster, prevalence = within a framing). |
| τ | $\tau$ (Prop 1); $\tau_{\epsilon/2}$ (Algorithm 1) | Not defined in text | In Setup, define "the shared threshold $\tau$" and in the algorithm write $\tau_{\epsilon/2}$ for its half-budget version. |
| k, k_max | $k_{\max}{=}30$ in Clustering error (Sec 4 today), defined only in Algorithm 1 (Sec 5) | Forward use | Fixed by moving the paragraph to Section 6. |
| Φ, $F_0$, $F_1$ | $F_y$ (generic), $\Phi$ (Prop 3) | Φ undefined | "where $\Phi$ is the standard normal distribution function". |
| α, a_f, b_f | α is steering strength (Setup), $a_f,b_f$ location and scale | No clash, but "a" appears again without subscript in Prop 2 | Write $a_f$ in Prop 2 for consistency. |
| s | probe score (Sec 4 and Algorithm 1) | OK |  |

Cross-references:

- "(Appendix, Table~\ref{tab:flood})" should be "Appendix~\ref{app:extra}, Table~\ref{tab:flood}".
- Method paragraph 1 ("The diagnosis and the model of Section~\ref{sec:theory} suggest ...") is fine.
- Prop 2 refers to Section 5 ("the assumption stated in Section 5"): the Method paragraph refers back to Proposition 2. Good, but both say "the assumption". After the rewrite, say the assumption once, in Prop 2, and refer to it by "the prevalence bound" in Section 5.
- Table 4 is cited from Prop 3 as `tab:breakdown`, but the table is placed right after the proposition at the cost of about 8 lines. See 5.
- Table 1 (`tab:detection`) is referred to in Prop 1 for "benign FPR falls to nearly zero", but that table shows only recall and AUROC (a point made in the v2 review that still stands). Add FPR to the table or say "(Appendix)".
- The section labelled "Section 3" in the text for Diagnosis is `sec:diagnosis`. Numbering after adding Section 4: Setup=3? In the PDF, Setup is Section 3, Diagnosis Section 4, Model Section 5, Method Section 6, Results Section 7, Discussion Section 8. Check every hard-coded "Section N" in the prose. I found none, only `\ref`, which is good; keep it so. In my rewrites above, "Section 4" etc. are placeholders, so use `Section~\ref{sec:theory}` in the final text.

---

## 5. Page budget

Measured on `main.pdf` (ACL review format): the Conclusion ends on page 8 and Limitations starts at the bottom of page 8, so the body fits, with very little slack (I estimate under 10 lines). Table 9 (Phi-3.5 replication, `table*`) is placed on page 9, after the Limitations heading. That table is cited from the body, so its placement risks a "body beyond 8 pages" objection. Moving the three paragraphs of section 2 to Section 6 is page-neutral; the following cuts create room (about 55 lines, or 0.6 page):

| # | Cut or move | Saves |
|---|---|---|
| 1 | Move Table 4 (`breakdown.tex`) to the appendix. Prop 3 already quotes its key cells (90% at π=0.05, 13% at π=0.30 for d=4; below 0.3% at π=0.5). Replace with "Table~\ref{tab:breakdown} (Appendix)". | about 9 lines |
| 2 | Move the Flooding numbers into the appendix table's caption plus one sentence in the body; keep only the fiction and code example. | 4 lines |
| 3 | Merge "Projecting out framing" and "Where to read" and shorten; the list of ranks "0, 4, 16 or 64 ... 54%, 54%, 54% and 48%" can be "54% for ranks 0 to 16 and 48% for rank 64". | 5 lines |
| 4 | Method paragraph 1: cut "Two facts help ..." to one sentence (Prop 2 now carries it). | 3 lines |
| 5 | Results, "A second model": drop the second paragraph's list of five ASR pairs to the two largest and refer to Table 9. | 5 lines |
| 6 | Results, "Generation": delete the sentences "Appendix gives the pre-registered configuration ... at 35% they fail." (duplicated in the appendix caption). | 3 lines |
| 7 | Discussion paragraph 1: replaced (3.6) by a shorter version. | 3 lines |
| 8 | Related Work "Representation-level defenses": cut the sentence on Kim and Lee's register conflation (cited elsewhere) and "Adaptive attacks ... recall to zero" can be merged with the sentence before. | 3 lines |
| 9 | Move Table 9 (Phi) to the appendix and keep the prose in the body. It has the numbers the paragraph cites, but the paragraph already states the ranges. | about 14 lines (table with caption) |
| 10 | Table 5 caption and Table 8 caption are long; trim each by one line. | 2 lines |

Items 1, 3, 4, 5, 7 and 9 together free about 0.6 page and are the least harmful. I would do 1, 4, 7 and either 9 or 5 before anything else. Do not cut the Setup "Protocol and its departures" paragraph; it is the honesty statement a reviewer checks first.

If Section 4 grows by the orienting paragraph (about 8 lines), the cuts above still leave a margin of roughly 0.4 page.

---

## 6. Smaller items

- Section 4 title: consider "Why the Threshold Moves" (describes what the section does for the reader). Current "A Model of the Shift" is accurate, and acceptable.
- Prop 3 text: "recall is below $0.3\%$ (Table~\ref{tab:breakdown})" for d≥2 at π=1/2: verify against the table, since z*=2.52 and d=2 gives m≈1, σ≈1.5 or so.
- Table 6 caption says "Correlation, and mean absolute error in percentage points." Say that the correlation is over the 26 points and that TPR and FPR are the two quantities compared.
- Diagnosis opening: "On withheld framings recall falls to 0--17% for every gate" refers to Table 1's four original withheld framings. Say "on the four original withheld framings"; the fresh framings include code and translation, where the shared threshold works (83% and 51% to 3% ASR).
- Setup: "Four were withheld from the start" and abstract "nine withheld": make sure the intro says "thirteen framings, four seen and nine withheld".
- Limitations first sentence "We study two small models (1.5B and 3.8B parameters)." fine.
- "Hybrid" and "adaptive" are used for both the rule and the (not-tested) attacker in the old draft; after the Flooding rewrite this is consistent: "adaptive" is the rule, "flooding attacker" or "attacker who shapes prompts" is the threat.
