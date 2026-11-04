# Related-work bibliography (compiled 2026-11-04)

BibTeX: `review/related_work.bib` (same keys).

Status legend:
- **[V]**: verified online in this session (title, authors, arXiv id, and venue where one is given).
- **[K]**: canonical paper with standard metadata, not re-fetched in this session. Spot-check before camera-ready.
- **[U]**: some detail could not be confirmed. The note says which.

Relation tags: **S** = supports our story, **C** = contrasts with or is a novelty threat to our story, **B** = baseline or evaluation resource, **M** = method precedent.

---

## 1. Jailbreak attacks (where the framings come from)

| Key | Citation | Link | Note |
|---|---|---|---|
| zou2023gcg | Zou, Wang, Carlini, Nasr, Kolter, Fredrikson. *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv 2023. [K] | https://arxiv.org/abs/2307.15043 | Introduces the GCG suffix attack and AdvBench. Most probes are calibrated on this canonical known framing. **B** |
| wei2023jailbroken | Wei, Haghtalab, Steinhardt. *Jailbroken: How Does LLM Safety Training Fail?* NeurIPS 2023. [K] | https://arxiv.org/abs/2307.02483 | Identifies competing objectives and mismatched generalisation, and shows base64 and other encodings bypass safety training. Gives the theoretical motivation for our "unseen framing" shift. **S** |
| liu2024autodan | Liu, Xu, Chen, Xiao. *AutoDAN.* ICLR 2024. [K] | https://arxiv.org/abs/2310.04451 | Genetic-algorithm generation of fluent jailbreaks. A standard attack family. **B** |
| chao2025pair | Chao, Robey, Dobriban, Hassani, Pappas, Wong. *Jailbreaking Black Box LLMs in Twenty Queries* (PAIR). SaTML 2025. [V; **U** venue] | https://arxiv.org/abs/2310.08419 | Attacker-LLM refinement that often produces roleplay or narrative framings. The arXiv record and authors are confirmed. The SaTML 2025 venue is from memory (the earlier version was a NeurIPS'23 R0-FoMo workshop paper). **B** |
| mehrotra2024tap | Mehrotra et al. *Tree of Attacks (TAP).* NeurIPS 2024. [K] | https://arxiv.org/abs/2312.02119 | Tree-search extension of PAIR. **B** |
| shen2024dan | Shen, Chen, Backes, Shen, Zhang. *"Do Anything Now".* CCS 2024. [K] | https://arxiv.org/abs/2308.03825 | Large-scale in-the-wild persona/DAN prompts. A source for persona framing. **B** |
| anil2024manyshot | Anil, Durmus, Panickssery, Sharma, et al. *Many-shot Jailbreaking.* NeurIPS 2024. [K] | NeurIPS 2024 proceedings (no arXiv) | Long-context in-context attack, another framing family. **B** |
| yuan2024cipherchat | Yuan et al. *GPT-4 Is Too Smart To Be Safe: Stealthy Chat via Cipher.* ICLR 2024. [K] | https://arxiv.org/abs/2308.06463 | Cipher and encoding attacks. A source for our encoding OOD split. **B** |
| greshake2023indirect | Greshake et al. *Not What You've Signed Up For: Indirect Prompt Injection.* AISec 2023. [K] | https://arxiv.org/abs/2302.12173 | Founding paper on prompt injection hidden in retrieved documents. A source for our injection framing. **B** |
| li2023deepinception | Li, Zhou, Zhu, Yao, Liu, Han. *DeepInception.* arXiv 2023. [V; no venue found] | https://arxiv.org/abs/2311.03191 | Nested-fiction jailbreak. The canonical fiction/roleplay attack. **B** |
| hughes2025bon | Hughes, Price, Lynch, et al. *Best-of-N Jailbreaking.* NeurIPS 2025. [V] | https://arxiv.org/abs/2412.03556 | Random augmentations (capitalisation, shuffling) beat circuit breakers. A recent attack showing that surface-form shift alone defeats representation defences. **S/B** |

## 2. Benchmarks

| Key | Citation | Link | Note |
|---|---|---|---|
| mazeika2024harmbench | Mazeika et al. *HarmBench.* ICML 2024. [K] | https://arxiv.org/abs/2402.04249 | Standard red-teaming benchmark. Its Llama-2-13B classifier serves as a judge. **B** |
| chao2024jailbreakbench | Chao et al. *JailbreakBench.* NeurIPS 2024 D&B. [K] | https://arxiv.org/abs/2404.01318 | JBB-Behaviors plus an artifact repository of attacks. **B** |
| rottger2024xstest | Röttger et al. *XSTest.* NAACL 2024. [K] | https://arxiv.org/abs/2308.01263 | Over-refusal suite. Use it to show recalibration does not inflate false positives on hard benign prompts. **B** |
| cui2025orbench | Cui, Chiang, Stoica, Hsieh. *OR-Bench.* ICML 2025. [V; **U** venue] | https://arxiv.org/abs/2405.20947 | 80k seemingly toxic benign prompts. The ICML 2025 acceptance is from memory and was not confirmed by search. **B** |
| souly2024strongreject | Souly et al. *A StrongREJECT for Empty Jailbreaks.* NeurIPS 2024 D&B. [K] | https://arxiv.org/abs/2402.10260 | Judge rubric that penalises empty jailbreaks. Use it for the generation-level evaluation. **B** |
| jiang2024wildteaming | Jiang et al. *WildTeaming at Scale* (WildJailbreak). NeurIPS 2024. [K] | https://arxiv.org/abs/2406.18510 | In-the-wild tactic mining, with adversarial-benign and adversarial-harmful splits. Useful because its benign prompts share jailbreak framing, which separates framing from harmfulness. **B/S** |

## 3. Representation-level safety, probes, and their robustness

| Key | Citation | Link | Note |
|---|---|---|---|
| zou2023repe | Zou et al. *Representation Engineering.* arXiv 2023. [K] | https://arxiv.org/abs/2310.01405 | Linear reading and control of concepts including harmfulness. The foundation for probe gates. **M** |
| arditi2024refusal | Arditi et al. *Refusal Is Mediated by a Single Direction.* NeurIPS 2024. [K] | https://arxiv.org/abs/2406.11717 | Refusal is a single direction read at post-instruction (template) tokens. Motivates our last-template-token readout. **M/S** |
| lee2025cast | Lee, Padhi, Natesan Ramamurthy, Miehling, Dognin, Nagireddy, Dhurandhar. *Programming Refusal with Conditional Activation Steering.* ICLR 2025 (spotlight). [V] | https://arxiv.org/abs/2409.05907 | CAST gates steering with a condition vector and a fixed similarity threshold. This is exactly the gate whose threshold we show shifts under unseen framings. **B (primary)** |
| zou2024circuitbreakers | Zou et al. *Improving Alignment and Robustness with Circuit Breakers.* NeurIPS 2024. [K] | https://arxiv.org/abs/2406.04313 | Representation rerouting defence, with harmfulness probes as baselines. BoN later bypasses it. **B** |
| zhao2025harmfulness | Zhao, Huang, Wu, Bau, Shi. *LLMs Encode Harmfulness and Refusal Separately.* arXiv 2025 (v4 Dec 2025). [V] | https://arxiv.org/abs/2507.11878 | Harmfulness is encoded at the last instruction token and refusal at the last sequence token. Some jailbreaks suppress refusal without flipping harmfulness. Its Latent Guard matches or beats Llama Guard 3. Directly motivates our multi-position readout and our "information preserved" claim. Check for a 2026 venue. **S (strong)** |
| zhang2025jbshield | Zhang et al. *JBShield.* USENIX Security 2025, pp. 8215–8234. [V] | https://arxiv.org/abs/2502.07557 | Toxic-concept plus jailbreak-concept activation detector, calibrated on a small reference set. This is the strongest activation-detector baseline, and it is calibrated on known attacks. **B** |
| mei2026hiddenguard | Mei, Liu, Wang, Bi, Yuan, Cheng. *HiddenGuard.* ACL 2026 (long). [V] | https://arxiv.org/abs/2410.02684 | Token-level representation router (PRISM) for in-stream redaction. **B** |
| hu2024gradientcuff | Hu, Chen, Ho. *Gradient Cuff.* NeurIPS 2024. [V] | https://arxiv.org/abs/2403.00867 | Refusal-loss landscape detector, evaluated on GCG, AutoDAN, PAIR, TAP, Base64, and LRL. A non-probe internal-signal baseline with a fixed threshold. **B** |
| xie2024gradsafe | Xie, Fang, Pi, Gong. *GradSafe.* ACL 2024. [V] | https://arxiv.org/abs/2402.13494 | Gradient-similarity detector on safety-critical parameters, with a fixed threshold. **B** |
| sheshadri2024lat | Sheshadri, ..., Casper. *Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors.* arXiv 2024. [V; **U** venue] | https://arxiv.org/abs/2407.15549 | Training-time latent robustness, which is complementary to our post-hoc recalibration. A NeurIPS 2024 listing exists, but whether it is main track or a workshop is unconfirmed, so cite it as arXiv. **C (orthogonal)** |
| bailey2026obfuscated | Bailey, Serrano, Sheshadri, et al. *Obfuscated Activations Bypass LLM Latent-Space Defenses.* ICLR 2026. [V] | https://arxiv.org/abs/2412.09565 | Adaptive attacks drive harmfulness-probe recall from 100% to 0%. This marks a limit of our method: recalibration addresses natural framing shift, not adaptive obfuscation. State this explicitly. **C** |
| mckenzie2025highstakes | McKenzie et al. *Detecting High-Stakes Interactions with Activation Probes.* NeurIPS 2025. [V] | https://arxiv.org/abs/2506.10805 | Probes generalise out of distribution in AUROC (>0.91), yet their recall at 1% FPR is modest. The same AUROC-versus-TPR@lowFPR gap we report, in a different domain. **S** |
| blandfort2025redteamprobes | Blandfort, Graham. *Red-teaming Activation Probes using Prompted LLMs.* arXiv 2025. [V] | https://arxiv.org/abs/2511.00554 | Black-box red-teaming finds style-driven false positives (legalese) and false negatives (bland procedural text) in probes. **S** |
| kirch2025features | Kirch, Weisser, Field, Yannakoudakis, Casper. *What Features in Prompts Jailbreak LLMs?* BlackboxNLP 2025. [V] | https://doi.org/10.18653/v1/2025.blackboxnlp-1.28 | 10.8k jailbreaks across 35 methods. Probe transfer is attack-family specific and jailbreak features look non-linear. Supports "framing-specific" behaviour. **S** |
| kim2026linearly | Kim, Lee. *Linearly Readable, Not Robust: How Linear Jailbreak Probes Conflate Benign Register with Adversarial Intent.* Findings of EMNLP 2026, pp. 35071–35088. [V] | https://aclanthology.org/2026.findings-emnlp.1648/ | **Closest prior work.** Probes are poorly calibrated and their near-perfect detection is threshold-dependent. Signal is partly benign register (style), and a short suffix kills detection. They propose LEACE erasure of a 3-D register subspace plus distribution-free (conformal) calibration that guarantees ≤5% FPR. Their calibration uses labelled benign data and one global threshold. **C (novelty threat)** |
| farzam2026hiding | Farzam, Behabahani, Malek, Nevmyvaka, Sapiro. *Hiding in Plain Text: Detecting Concealed Jailbreaks via Activation Disentanglement* (ReDAct / FrameShield / GoalFrameBench). arXiv 2026. [V] | https://arxiv.org/abs/2602.19396 | Disentangles *goal* and *framing* factors in activations and runs anomaly detection on the framing representation. Very close conceptually: it also treats framing as a nuisance factor, but handles it by representation learning, not threshold recalibration. **C (novelty threat)** |
| luo2026measuring | Luo et al. *Measuring the Wrong Thing: Internal Harmfulness Scores Anti-Rank Successful Jailbreaks.* arXiv 2026. [V] | https://arxiv.org/abs/2608.09624 | Under wrapping, intent AUROC falls 0.936→0.803. States that "distribution shift degrades calibration and threshold transfer before it degrades ranking", which is nearly our headline claim. Also reports that benign-quantile thresholds become over-conservative under benign shift. **C/S (novelty threat on the diagnosis)** |
| schwarz2026entanglement | Schwarz. *The Entanglement Wall: Activation-Space Probes as Risk Detectors, Not Context Adjudicators.* arXiv 2026. [V] | https://arxiv.org/abs/2607.13075 | Transferred thresholds catch harmful requests but block many matched benign ones. Refitting thresholds does not fix this on topic-matched pairs. Contrasts with us: there the information is missing, while we claim it is retained. **C** |
| gao2026alarm | Gao. *From Probe Scores to Alarm Policies.* DAI 2026. [V] | https://arxiv.org/abs/2610.04575 | AUROC up to 0.957, yet low recall at locked 5–10% false-alarm budgets. A 10% threshold realises 18.7–20%. Argues for operating-point evaluation. **S** |
| david2026prompted | David, Fomin. *Prompted to Discriminate: Generalizing Malicious-Input Probes in the Wild.* NeurIPS 2026 workshop. [V] | https://arxiv.org/abs/2610.02413 | Leave-one-dataset-out over 13 benchmarks. Classification-style suffixes add up to 4 AUC OOD and transfer to multi-position probes. Close to our LOFO and multi-position setup, but it targets the representation, not the threshold. **C/B** |
| azov2026sallie | Azov, Rivlin, Shtar. *SALLIE.* arXiv 2026. [V] | https://arxiv.org/abs/2604.06247 | Layer-wise kNN probes on last-token activations. Compares shared, threshold-only, and full calibration, and finds threshold-only recalibration often suffices. Operating points are "distribution-dependent". Supports our framing as a threshold problem, but its recalibration appears to use labelled target data. **S/C** |
| piet2025jailbreaksovertime | Piet et al. *JailbreaksOverTime: Detecting Jailbreak Attacks Under Distribution Shift.* arXiv 2025 (ACM DOI 10.1145/3733799.3762981). [V; **U** venue name] | https://arxiv.org/abs/2504.19440 | Detectors degrade over 10 months of drift. Weekly label-free self-training restores FNR to 0.3% at 0.1% FPR. **Closest label-free adaptation precedent**, but for text classifiers and with retraining, not per-cluster threshold standardisation. **C/M** |
| sharma2025constitutional | Sharma et al. *Constitutional Classifiers.* arXiv 2025. [K] | https://arxiv.org/abs/2501.18837 | Industrial input/output classifiers that withstood extensive red-teaming. Context for deployed gates. **B** |

## 4. Guard models and judges

| Key | Citation | Link | Note |
|---|---|---|---|
| inan2023llamaguard | Inan et al. *Llama Guard.* arXiv 2023. [K] | https://arxiv.org/abs/2312.06674 | Llama Guard 1. Versions 2 and 3 have only model cards, so cite dubey2024llama3 for Guard 3. **B** |
| dubey2024llama3 | Llama Team. *The Llama 3 Herd of Models.* arXiv 2024. [K] | https://arxiv.org/abs/2407.21783 | Describes Llama Guard 3. **B** |
| han2024wildguard | Han et al. *WildGuard.* NeurIPS 2024 D&B. [K] | https://arxiv.org/abs/2406.18495 | Open moderation model covering harm, jailbreak, and refusal. **B** |
| zeng2024shieldgemma | Zeng et al. *ShieldGemma.* arXiv 2024. [K] | https://arxiv.org/abs/2407.21772 | Gemma-based guard. **B** |
| qwen2025qwen3guard | Qwen Team (Zhao et al.). *Qwen3Guard Technical Report.* arXiv 2025. [V] | https://arxiv.org/abs/2510.14276 | Generative 3-way (safe/controversial/unsafe) guard plus a streaming token-level guard, 119 languages. Has a strict mode and a loose mode, which are effectively two thresholds. **B** |
| padhi2025graniteguardian | Padhi et al. *Granite Guardian.* NAACL 2025 Industry, pp. 607–615. [V] | https://arxiv.org/abs/2412.07724 | IBM guard family (the same group as CAST). **B** |
| mazeika2024harmbench | (see §2) | | HarmBench classifier as the generation judge. **B** |

## 5. Calibration under shift, test-time adaptation, and robust statistics

| Key | Citation | Link | Note |
|---|---|---|---|
| saerens2002adjusting | Saerens, Latinne, Decaestecker. *Neural Computation* 14(1), 2002. [K] | — | EM prior re-estimation on unlabelled data. The classic label-shift correction. **M** |
| lipton2018bbse | Lipton, Wang, Smola. *BBSE.* ICML 2018. [K] | https://arxiv.org/abs/1802.03916 | Black-box label-shift estimation. Contrast: it assumes p(x given y) is fixed, which framing shift violates. **M/C** |
| alexandari2020labelshift | Alexandari, Kundaje, Shrikumar. ICML 2020. [K] | https://arxiv.org/abs/1901.06852 | Bias-corrected calibration plus EM is a strong label-shift baseline. **M** |
| wang2021tent | Wang et al. *Tent.* ICLR 2021. [K] | https://arxiv.org/abs/2006.10726 | Fully test-time adaptation by entropy minimisation. Ours is a parameter-free TTA analogue acting on the score scale. **M** |
| tibshirani2019conformalshift | Tibshirani, Barber, Candès, Ramdas. NeurIPS 2019. [K] | https://arxiv.org/abs/1904.06019 | Weighted conformal under covariate shift. FPR guarantees need a known or estimable likelihood ratio. **M/C** |
| podkopaev2021labelshiftconformal | Podkopaev, Ramdas. UAI 2021. [K] | https://arxiv.org/abs/2103.03323 | Conformal classification under label shift. **M** |
| hebertjohnson2018multicalibration | Hébert-Johnson, Kim, Reingold, Rothblum. *Multicalibration.* ICML 2018. [K] | https://arxiv.org/abs/1711.08513 | Calibration on every identifiable subgroup. The conceptual ancestor of per-cluster calibration, but it requires labels. **M** |
| ziegler2019unsupervised | Ziegler, Czyż. *Unsupervised Recalibration.* arXiv 2019. [V] | https://arxiv.org/abs/1908.09157 | Label-free recalibration from predictions alone, applied per subpopulation. The closest generic-ML precedent for "label-free, cluster-wise recalibration". Cite it so reviewers do not think we claim the generic idea. **M (novelty context)** |
| rousseeuw1993mad | Rousseeuw, Croux. *JASA* 88(424), 1993. [K] | — | MAD and robust scale estimators (Sn, Qn). **M** |
| leys2013mad | Leys et al. *JESP* 49(4), 2013. [K] | https://doi.org/10.1016/j.jesp.2013.03.013 | Practical case for median/MAD standardisation. **M** |
| hendrycks2017baseline | Hendrycks, Gimpel. ICLR 2017. [K] | https://arxiv.org/abs/1610.02136 | Max-softmax OOD baseline. OOD thresholds are typically fixed at 95% TPR on in-distribution data. **M** |
| liu2020energy | Liu, Wang, Owens, Li. *Energy-based OOD Detection.* NeurIPS 2020. [K] | https://arxiv.org/abs/2010.03057 | Energy scores. Threshold transfer is a known OOD pain point. **M** |

Not verified or deliberately omitted: no "HiddenGuard-like" paper or "linear probes for harmful intent" paper was found under those exact names beyond the entries above. Specific 2026 roleplay-attack papers were not verified individually, so DeepInception, PAIR, and WildJailbreak stand in for fiction/roleplay framings.

---

## Closest prior work and how our contribution differs

1. **Kim & Lee (Findings EMNLP 2026)** is the most serious threat. It already argues that linear jailbreak probes are *poorly calibrated*, that their near-perfect detection is *threshold-dependent*, and that a fix needs *calibration* (distribution-free, ≤5% FPR) plus erasure of a benign-register subspace. Our differentiators are:
   - We diagnose collapse under *unseen attack framings* as a ranking-preserving score shift (AUROC 0.98–0.99 against TPR@1%FPR of 5–17%).
   - Our recalibration is *label-free and per-cluster* on the live stream, not a single global conformal threshold fit on labelled benign data.
   - We apply it to *gates for conditional steering* (CAST), not only to filters.

   We must cite this paper prominently and compare against conformal calibration as a baseline.
2. **Luo et al. (arXiv Aug 2026)** states almost verbatim that "shift degrades calibration and threshold transfer before ranking". This weakens any claim that the diagnosis itself is new. Frame our diagnosis as systematic confirmation across framings, gates, and readout positions, and put the novelty in the remedy.
3. **Farzam et al. (FrameShield, arXiv Feb 2026)** treats *framing* as a nuisance factor to disentangle from the goal. It is the same intuition, addressed through learned representations. Ours needs no training.
4. **Piet et al. (JailbreaksOverTime, 2025)** shows label-free adaptation (self-training) of jailbreak detectors under drift. Ours avoids retraining and needs no pseudo-labels.
5. **Ziegler & Czyż (2019)** gives the generic label-free, per-subpopulation recalibration, and multicalibration covers group-wise calibration. The statistical idea is not new. The contribution lies in applying it to activation safety gates, choosing the framing features to cluster on, and the empirical evidence.

Supporting context: SALLIE, McKenzie et al., and Gao report AUROC-versus-low-FPR gaps and the sufficiency of threshold-only recalibration. Zhao et al. support the multi-position readout. Bailey et al. bound the claim, since adaptive obfuscation is out of scope.
