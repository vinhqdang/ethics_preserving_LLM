# Literature and Venue-Compliance Audit (ACL 2027 Main), 2026-11-02

Method: arXiv API (export.arxiv.org) queried by ID for every bib entry and every candidate paper below. Authors/titles are as returned by arXiv. Venue info comes from arXiv comment/journal_ref fields where present; items marked (venue: unconfirmed) were not confirmable from arXiv metadata and are given as arXiv-only (or from my prior knowledge, flagged). ACL/ARR rules are from https://aclrollingreview.org/cfp (fetched) and an aggregator report on ACL 2027 (the official 2027.aclweb.org call URL returned 404; re-check it).

## 1. Reference verification

All 29 bib entries describe real papers. However, 9 entries have wrong/hallucinated author lists, 1 has a wrong arXiv ID, 1 has a wrong title, and several have outdated venues. Cite-key audit: every `\citep/\citet` key in main.tex exists in the bib. The one bib entry never cited is `hendrycks2021measuring` (MMLU); cite it (if MMLU is used in the Qwen evaluation) or delete it.

### Errors (must fix)

| key | problem | correction |
|---|---|---|
| zou2023representation | Authors wrong: "Mantena, Mantas" and "Chiu, Peter" do not exist (fabricated). Real list has Mazeika, Dombrowski, Goel, Li, Byun, Wang, Mallen, Basart, Koyejo, Song, Fredrikson, Kolter | see corrected bib |
| zou2024circuit | Title outdated (now "Improving Alignment and Robustness with Circuit Breakers", NeurIPS 2024); author "Du, Yacine" fabricated | see bib |
| arditi2024refusal | arXiv ID wrong (2406.11707 is a LiDAR trajectory-attack paper; correct is 2406.11717); authors wrong ("Obeng", "Akyürek" fabricated; correct Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda). NeurIPS 2024 | see bib |
| inan2023llama | Authors wrong ("Zheng, Mortazavi, Bansal, Smith" fabricated; real: Mao, Tontchev, Hu, Fuller, Testuggine, Khabsa) | see bib |
| chao2024jailbreakbench | Authors wrong (real list includes Debenedetti, Andriushchenko, Croce, Sehwag, Flammarion, Tramer; Dobriban etc.). Venue: NeurIPS 2024 Datasets and Benchmarks | see bib |
| rottger2023xstest | Authors wrong (real: Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy). Title wrong ("Exaggerated Safety Behaviours in Large Language Models"). NAACL 2024 Main | see bib |
| cui2024orbench | Authors wrong (real: Justin Cui, Wei-Lin Chiang, Ion Stoica, Cho-Jui Hsieh). ICML 2025 | see bib |
| carlini2019evaluating | Authors wrong ("Jonas, Jonas" -> Rauber, Tsipras, Kurakin missing); @inproceedings with booktitle=arXiv is malformed | see bib |
| ji2023beavertails | Title is not the arXiv/NeurIPS title ("...of LLM via a Human-Preference Dataset"); authors differ from arXiv (Chi Zhang, Ce Bian, Ruiyang Sun...); pages 68123-68145 unverified/likely wrong. NeurIPS 2023 Datasets and Benchmarks | see bib |
| ji2025saferlhfv | Author list truncated/garbled (last author is Sirui Han, not Han Sirui Guo Yike Yang Yaodong; arXiv lists 14 authors ending Yida Tang, Sirui Han). Venue not confirmed by arXiv; keep arXiv | see bib |

### Outdated venues / type fixes (should fix)

- ji2024pkusaferlhf: ACL 2025 Main (arXiv comment). Cite as inproceedings.
- lee2024programming (CAST): ICLR 2025 (Spotlight).
- zou2024circuit: NeurIPS 2024. arditi2024refusal: NeurIPS 2024. robey2023smoothllm: arXiv 2310.03684 only in metadata; the bib says TMLR (published in TMLR 2025 per my knowledge; unconfirmed, keep or revert to arXiv).
- chao2023jailbreaking (PAIR): published at IEEE SaTML 2025 (my knowledge, unconfirmed); arXiv acceptable.
- russinovich2024crescendo: USENIX Security 2025 (my knowledge, unconfirmed).
- zou2023universal: arXiv only; fine.
- wei2024jailbroken: key says 2024 but it is NeurIPS 2023 (year field correct); use @inproceedings.
- christiano2017deep, ouyang2022training, tramer2020adaptive: @article with conference `journal`; switch to @inproceedings (cosmetic).
- mazeika2024harmbench: ICML 2024 is right (PMLR 235).

Verified correct as given: ouyang2022, christiano2017, dai2024safe, rafailov2023, bai2022, absil2008, bronstein2021, hendrycks2021 (ICLR 2021), liu2023autodan (ICLR 2024), xu2024safedecoding (ACL 2024), tramer2020adaptive (NeurIPS 33, pp. 1633-1645), zou2023universal.

### Corrected bib entries

```bibtex
@inproceedings{zou2023representation,
  title     = {Representation Engineering: A Top-Down Approach to {AI} Transparency},
  author    = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and Dombrowski, Ann-Kathrin and Goel, Shashwat and Li, Nathaniel and Byun, Michael J. and Wang, Zifan and Mallen, Alex and Basart, Steven and Koyejo, Sanmi and Song, Dawn and Fredrikson, Matt and Kolter, J. Zico and Hendrycks, Dan},
  booktitle = {arXiv preprint arXiv:2310.01405},
  year      = {2023}
}
@inproceedings{zou2024circuit,
  title     = {Improving Alignment and Robustness with Circuit Breakers},
  author    = {Zou, Andy and Phan, Long and Wang, Justin and Duenas, Derek and Lin, Maxwell and Andriushchenko, Maksym and Wang, Rowan and Kolter, J. Zico and Fredrikson, Matt and Hendrycks, Dan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  note      = {arXiv:2406.04313}
}
@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  note      = {arXiv:2406.11717}
}
@article{inan2023llama,
  title   = {Llama Guard: {LLM}-based Input-Output Safeguard for Human-{AI} Conversations},
  author  = {Inan, Hakan and Upasani, Kartikeya and Chi, Jianfeng and Rungta, Rashi and Iyer, Krithika and Mao, Yuning and Tontchev, Michael and Hu, Qing and Fuller, Brian and Testuggine, Davide and Khabsa, Madian},
  journal = {arXiv preprint arXiv:2312.06674},
  year    = {2023}
}
@inproceedings{chao2024jailbreakbench,
  title     = {{JailbreakBench}: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author    = {Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym and Croce, Francesco and Sehwag, Vikash and Dobriban, Edgar and Flammarion, Nicolas and Pappas, George J. and Tram{\`e}r, Florian and Hassani, Hamed and Wong, Eric},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume    = {37},
  year      = {2024},
  note      = {arXiv:2404.01318}
}
@inproceedings{rottger2023xstest,
  title     = {{XSTest}: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models},
  author    = {R{\"o}ttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  year      = {2024},
  note      = {arXiv:2308.01263}
}
@inproceedings{cui2024orbench,
  title     = {{OR-Bench}: An Over-Refusal Benchmark for Large Language Models},
  author    = {Cui, Justin and Chiang, Wei-Lin and Stoica, Ion and Hsieh, Cho-Jui},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2405.20947}
}
@article{carlini2019evaluating,
  title   = {On Evaluating Adversarial Robustness},
  author  = {Carlini, Nicholas and Athalye, Anish and Papernot, Nicolas and Brendel, Wieland and Rauber, Jonas and Tsipras, Dimitris and Goodfellow, Ian and Madry, Aleksander and Kurakin, Alexey},
  journal = {arXiv preprint arXiv:1902.06705},
  year    = {2019}
}
@inproceedings{ji2023beavertails,
  title     = {{BeaverTails}: Towards Improved Safety Alignment of {LLM} via a Human-Preference Dataset},
  author    = {Ji, Jiaming and Liu, Mickel and Dai, Juntao and Pan, Xuehai and Zhang, Chi and Bian, Ce and Zhang, Chi and Sun, Ruiyang and Wang, Yizhou and Yang, Yaodong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  volume    = {36},
  year      = {2023},
  note      = {arXiv:2307.04657}
}
@inproceedings{ji2024pkusaferlhf,
  title     = {{PKU-SafeRLHF}: Towards Multi-Level Safety Alignment for {LLMs} with Human Preference},
  author    = {Ji, Jiaming and Hong, Donghai and Zhang, Borong and Chen, Boyuan and Dai, Juntao and Zheng, Boren and Qiu, Tianyi and Zhou, Jiayi and Wang, Kaile and Li, Boxuan and Han, Sirui and Guo, Yike and Yang, Yaodong},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  note      = {arXiv:2406.15513}
}
@article{ji2025saferlhfv,
  title   = {Safe {RLHF-V}: Safe Reinforcement Learning from Multi-modal Human Feedback},
  author  = {Ji, Jiaming and Chen, Xinyu and Pan, Rui and Zhang, Conghui and Zhu, Han and Li, Jiahao and Hong, Donghai and Chen, Boyuan and Zhou, Jiayi and Wang, Kaile and Dai, Juntao and Chan, Chi-Min and Tang, Yida and Han, Sirui and others},
  journal = {arXiv preprint arXiv:2503.17682},
  year    = {2025}
}
@inproceedings{lee2024programming,
  title     = {Programming Refusal with Conditional Activation Steering},
  author    = {Lee, Bruce W. and Padhi, Inkit and Ramamurthy, Karthikeyan Natesan and Miehling, Erik and Dognin, Pierre and Nagireddy, Manish and Dhurandhar, Amit},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {Spotlight. arXiv:2409.05907}
}
```

Also fix: `ji2025saferlhfv` is cited with a method claim; check the text refers to it as a multimodal method only.

## 2. Missing related work (prioritised; all IDs and author lists verified via arXiv API)

Priority A (a reviewer will expect these; ARETE is a refusal-geometry/steering paper with adaptive-attack claims):
1. Nasr et al. 2025, The Attacker Moves Second (adaptive attacks break 12 defenses). Directly undermines a 2.82% PGD-only claim.
2. Wollschläger et al. 2025, Geometry of Refusal: concept cones (refusal is multi-dimensional, contradicts a single-direction view).
3. Pan et al. 2025, Hidden Dimensions of LLM Alignment (orthogonal safety directions).
4. Zhao et al. 2025, LLMs Encode Harmfulness and Refusal Separately (closest to the intent/context decomposition idea).
5. Sheshadri et al. 2024, Latent Adversarial Training (latent-space robustness baseline).
6. Yu et al. 2024, ReFAT (refusal-feature adversarial training).
7. Yousefpour et al. 2025, RepBend (ACL 2025 main; representation-level safety baseline).
8. Andriushchenko et al., simple adaptive attacks (ICLR 2025).
9. Tan et al., reliability/generalisation of steering vectors (NeurIPS 2024).
10. Wu et al., AxBench (steering baselines vs SAEs).
11. Souly et al., StrongREJECT (ASR metric validity).
12. Qi et al., shallow safety alignment (ICLR 2025).

Priority B:
13. Rosati et al., Representation Noising (NeurIPS 2024).
14. Marshall et al., Refusal is an affine function.
15. O'Brien et al., refusal steering with SAEs.
16. Panickssery/Rimsky et al., CAA (ACL 2024) and Turner et al., ActAdd.
17. Mehrotra et al., TAP (NeurIPS 2024).
18. Hughes et al., Best-of-N Jailbreaking.
19. Anil et al., Many-shot Jailbreaking (NeurIPS 2024; no arXiv paper).
20. An et al., PHTest (COLM 2024).
21. Sharma et al., Constitutional Classifiers.
22. Hu et al., Gradient Cuff (NeurIPS 2024).

Priority C (geometry/invariance):
23. Arjovsky et al., IRM. 24. Higgins et al., definition of disentangled representations. 25. Davidson et al., hyperspherical VAE (UAI 2018). 26. Wang and Isola, alignment/uniformity on the hypersphere (ICML 2020).

```bibtex
@article{nasr2025attacker,
  title   = {The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against {LLM} Jailbreaks and Prompt Injections},
  author  = {Nasr, Milad and Carlini, Nicholas and Sitawarin, Chawin and Schulhoff, Sander V. and Hayes, Jamie and Ilie, Michael and Pluto, Juliette and Song, Shuang and Chaudhari, Harsh and Shumailov, Ilia and Thakurta, Abhradeep and Xiao, Kai Yuanqing and Terzis, Andreas and Tram{\`e}r, Florian},
  journal = {arXiv preprint arXiv:2510.09023},
  year    = {2025}
}
@inproceedings{wollschlager2025geometry,
  title     = {The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence},
  author    = {Wollschl{\"a}ger, Tom and Elstner, Jannes and Geisler, Simon and Cohen-Addad, Vincent and G{\"u}nnemann, Stephan and Gasteiger, Johannes},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2502.17420; venue from prior knowledge, unconfirmed by arXiv metadata}
}
@inproceedings{pan2025hidden,
  title     = {The Hidden Dimensions of {LLM} Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions},
  author    = {Pan, Wenbo and Liu, Zhichao and Chen, Qiguang and Zhou, Xiangyang and Yu, Haining and Jia, Xiaohua},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2502.09674}
}
@article{zhao2025harmfulness,
  title   = {{LLMs} Encode Harmfulness and Refusal Separately},
  author  = {Zhao, Jiachen and Huang, Jing and Wu, Zhengxuan and Bau, David and Shi, Weiyan},
  journal = {arXiv preprint arXiv:2507.11878},
  year    = {2025}
}
@article{sheshadri2024latent,
  title   = {Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in {LLMs}},
  author  = {Sheshadri, Abhay and Ewart, Aidan and Guo, Phillip and Lynch, Aengus and Wu, Cindy and Hebbar, Vivek and Sleight, Henry and Stickland, Asa Cooper and Perez, Ethan and Hadfield-Menell, Dylan and Casper, Stephen},
  journal = {arXiv preprint arXiv:2407.15549},
  year    = {2024}
}
@article{yu2024refat,
  title   = {Robust {LLM} Safeguarding via Refusal Feature Adversarial Training},
  author  = {Yu, Lei and Do, Virginie and Hambardzumyan, Karen and Cancedda, Nicola},
  journal = {arXiv preprint arXiv:2409.20089},
  year    = {2024}
}
@inproceedings{yousefpour2025repbend,
  title     = {Representation Bending for Large Language Model Safety},
  author    = {Yousefpour, Ashkan and Kim, Taeheon and Kwon, Ryan S. and Lee, Seungbeen and Jeung, Wonje and Han, Seungju and Wan, Alvin and Ngan, Harrison and Yu, Youngjae and Choi, Jonghyun},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025},
  note      = {arXiv:2504.01550}
}
@inproceedings{andriushchenko2025jailbreaking,
  title     = {Jailbreaking Leading Safety-Aligned {LLMs} with Simple Adaptive Attacks},
  author    = {Andriushchenko, Maksym and Croce, Francesco and Flammarion, Nicolas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2404.02151}
}
@inproceedings{tan2024steering,
  title     = {Analysing the Generalisation and Reliability of Steering Vectors},
  author    = {Tan, Daniel and Chanin, David and Lynch, Aengus and Kanoulas, Dimitrios and Paige, Brooks and Garriga-Alonso, Adria and Kirk, Robert},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2407.12404 (arXiv title: Analyzing the Generalization and Reliability of Steering Vectors); venue from prior knowledge}
}
@inproceedings{wu2025axbench,
  title     = {{AxBench}: Steering {LLMs}? Even Simple Baselines Outperform Sparse Autoencoders},
  author    = {Wu, Zhengxuan and Arora, Aryaman and Geiger, Atticus and Wang, Zheng and Huang, Jing and Jurafsky, Dan and Manning, Christopher D. and Potts, Christopher},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {arXiv:2501.17148; venue from prior knowledge}
}
@inproceedings{souly2024strongreject,
  title     = {A {StrongREJECT} for Empty Jailbreaks},
  author    = {Souly, Alexandra and Lu, Qingyuan and Bowen, Dillon and Trinh, Tu and Hsieh, Elvis and Pandey, Sana and Abbeel, Pieter and Svegliato, Justin and Emmons, Scott and Watkins, Olivia and Toyer, Sam},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2024},
  note      = {arXiv:2402.10260; venue from prior knowledge}
}
@inproceedings{qi2025shallow,
  title     = {Safety Alignment Should Be Made More Than Just a Few Tokens Deep},
  author    = {Qi, Xiangyu and Panda, Ashwinee and Lyu, Kaifeng and Ma, Xiao and Roy, Subhrajit and Beirami, Ahmad and Mittal, Prateek and Henderson, Peter},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2406.05946}
}
@inproceedings{rosati2024representation,
  title     = {Representation Noising: A Defence Mechanism Against Harmful Finetuning},
  author    = {Rosati, Domenic and Wehner, Jan and Williams, Kai and Bartoszcze, {\L}ukasz and Atanasov, David and Gonzales, Robie and Majumdar, Subhabrata and Maple, Carsten and Sajjad, Hassan and Rudzicz, Frank},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2405.14577}
}
@article{marshall2024affine,
  title   = {Refusal in {LLMs} is an Affine Function},
  author  = {Marshall, Thomas and Scherlis, Adam and Belrose, Nora},
  journal = {arXiv preprint arXiv:2411.09003},
  year    = {2024}
}
@article{obrien2024sae,
  title   = {Steering Language Model Refusal with Sparse Autoencoders},
  author  = {O'Brien, Kyle and Majercak, David and Fernandes, Xavier and Edgar, Richard and Bullwinkel, Blake and Chen, Jingya and Nori, Harsha and Carignan, Dean and Horvitz, Eric and Poursabzi-Sangdeh, Forough},
  journal = {arXiv preprint arXiv:2411.11296},
  year    = {2024}
}
@inproceedings{rimsky2024steering,
  title     = {Steering {Llama} 2 via Contrastive Activation Addition},
  author    = {Panickssery, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024},
  note      = {arXiv:2312.06681; venue from prior knowledge}
}
@article{turner2023activation,
  title   = {Steering Language Models With Activation Engineering},
  author  = {Turner, Alexander Matt and Thiergart, Lisa and Leech, Gavin and Udell, David and Vazquez, Juan J. and Mini, Ulisse and MacDiarmid, Monte},
  journal = {arXiv preprint arXiv:2308.10248},
  year    = {2023}
}
@inproceedings{mehrotra2024tap,
  title     = {Tree of Attacks: Jailbreaking Black-Box {LLMs} Automatically},
  author    = {Mehrotra, Anay and Zampetakis, Manolis and Kassianik, Paul and Nelson, Blaine and Anderson, Hyrum and Singer, Yaron and Karbasi, Amin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2312.02119}
}
@article{hughes2024bon,
  title   = {Best-of-{N} Jailbreaking},
  author  = {Hughes, John and Price, Sara and Lynch, Aengus and Schaeffer, Rylan and Barez, Fazl and Koyejo, Sanmi and Sleight, Henry and Jones, Erik and Perez, Ethan and Sharma, Mrinank},
  journal = {arXiv preprint arXiv:2412.03556},
  year    = {2024}
}
@inproceedings{anil2024manyshot,
  title     = {Many-shot Jailbreaking},
  author    = {Anil, Cem and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  note      = {No arXiv version; NOT verified via arXiv. Fill full author list from the NeurIPS proceedings page before use}
}
@inproceedings{an2024phtest,
  title     = {Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models},
  author    = {An, Bang and Zhu, Sicheng and Zhang, Ruiyi and Panaitescu-Liess, Michael-Andrei and Xu, Yuancheng and Huang, Furong},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2024},
  note      = {arXiv:2409.00598; venue from prior knowledge}
}
@article{sharma2025constitutional,
  title   = {Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming},
  author  = {Sharma, Mrinank and Tong, Meg and Mu, Jesse and Wei, Jerry and Kruthoff, Jorrit and Goodfriend, Scott and Ong, Euan and Peng, Alwin and Agarwal, Raj and Anil, Cem and Askell, Amanda and Bailey, Nathan and Benton, Joe and others},
  journal = {arXiv preprint arXiv:2501.18837},
  year    = {2025}
}
@inproceedings{hu2024gradientcuff,
  title     = {Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes},
  author    = {Hu, Xiaomeng and Chen, Pin-Yu and Ho, Tsung-Yi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  note      = {arXiv:2403.00867}
}
@article{arjovsky2019invariant,
  title   = {Invariant Risk Minimization},
  author  = {Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
  journal = {arXiv preprint arXiv:1907.02893},
  year    = {2019}
}
@article{higgins2018disentangled,
  title   = {Towards a Definition of Disentangled Representations},
  author  = {Higgins, Irina and Amos, David and Pfau, David and Racaniere, Sebastien and Matthey, Loic and Rezende, Danilo and Lerchner, Alexander},
  journal = {arXiv preprint arXiv:1812.02230},
  year    = {2018}
}
@inproceedings{davidson2018hyperspherical,
  title     = {Hyperspherical Variational Auto-Encoders},
  author    = {Davidson, Tim R. and Falorsi, Luca and De Cao, Nicola and Kipf, Thomas and Tomczak, Jakub M.},
  booktitle = {Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence (UAI)},
  pages     = {856--865},
  year      = {2018}
}
@inproceedings{wang2020alignment,
  title     = {Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author    = {Wang, Tongzhou and Isola, Phillip},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  year      = {2020},
  note      = {arXiv:2005.10242}
}
```

Not included because I could not verify: "Jailbreak Antidote" (ID not found), PHTest/Many-shot full metadata beyond what is stated, a 2026 refusal-geometry paper (arXiv keyword search for 2025-2026 returned nothing relevant; do a manual Google Scholar / ACL Anthology pass for ACL 2026 and ICLR 2026 refusal-direction and steering-defense papers before submission, since the arXiv API search is weak). Venues marked "from prior knowledge" should be spot-checked.

Where to cite in text: Related Work "Representation Engineering and Activation Steering" paragraph should add A1-A10; the Experiments baselines should include at least CAST, circuit breakers, and ideally RepBend/ReFAT/LAT (or justify omission, as these are the direct competitors); adaptive-attack paragraph must cite Nasr et al., Carlini 2019, Tramer 2020, Andriushchenko 2025.

## 3. ACL 2027 compliance

Sources: ARR CFP (aclrollingreview.org/cfp, fetched 2026-11-02). Aggregator report (aiweekly.co) on ACL 2027: ARR deadline Jan 4 2027, conference Aug 20-22 2027, long = 8 pages (9 camera-ready), special theme "homogenization and knowledge collapse in LLMs" (theme is optional for main track). Official ACL 2027 call page was not retrievable (404); confirm dates and page limit at the official site.

Requirements:
- Template: official ACL style files from the acl-org GitHub repo (acl.sty, acl_natbib.bst), review mode. Modified styles, other-conference templates, or non-conforming paper size/margins/fonts are rejected without review.
- Length: long paper 8 pages of content; Limitations (mandatory, desk-reject if missing, may not add new results), optional Ethical Considerations, references, and appendices are outside the limit. Essential derivations/pseudocode must be in the main text, not only the appendix.
- Anonymity: no names/affiliations, no identifying self-citations, anonymised code/data links (no Dropbox-type links); preprint status declared in the submission form.
- Responsible NLP Research Checklist: mandatory in OpenReview/ARR form; incorrect answers risk desk rejection. Must cover compute budget, licences of artifacts (PKU-SafeRLHF, Qwen2.5), seeds/variance, and AI-assistance disclosure.
- AI assistance: AI cannot be an author; writing/coding assistance disclosed in the checklist and Acknowledgements (README for code). Literature-search tools allowed but the authors own citation accuracy. Note that fabricated author lists, as found in Section 1, would be a policy problem under this clause.
- Reproducibility: code and data as anonymous .zip/.tgz supplementary; report seeds, hyperparameters, and confidence intervals.

Deviations in main.tex:
1. `\documentclass[11pt,twocolumn,a4paper]{article}` with custom `geometry` and `times`: must be replaced by `\usepackage[review]{acl}` with the ACL template (Times, A4, 2.5cm margins, two-column 11pt). Remove custom geometry, `microtype` conflicts and custom `hypersetup` colours.
2. Custom `\title`/`\author`/`\maketitle` with "Anonymous ACL Submission / ACL 2027 Main Conference Track": use template default (review mode prints "Anonymous ACL submission" itself).
3. Bibliography: `plainnat` + `[numbers]` natbib; ACL requires `\bibliographystyle{acl_natbib}` with author-year citations (\citep/\citet already used, good).
4. `figure*` TikZ with `\resizebox{0.96\textwidth}` and `table*` spanning pages: check these still fit 8 pages once margins change; the current build is reportedly 10 pages including limitations, references, so main body may exceed 8. Verify by compiling with acl.sty; trim or move non-essential tables.
5. Limitations present and positioned before references (good) but is a `\section*` after Conclusion with 3 items; broaden (Qwen2.5-1.5B only, one dataset taxonomy, synthetic attack families, no human eval, over-refusal on a handful of benign sets, white-box requirement, PGD adaptive attack is not an optimal adaptive attack per Nasr et al.).
6. Ethical Considerations present; add dual-use discussion (published steering barriers also reveal what to evade), data licences, and the AI-assistance statement in Acknowledgements (the Acknowledgements section is absent; for review submission it should be omitted or anonymised, but disclosure goes in the checklist).
7. Headings use `\&` and bold-everything in headings/abstract; ACL style is plain.
8. Abstract: ACL abstracts are one paragraph, roughly 150-200 words; current one has two paragraphs and about 330 words.
9. Appendix: none exists. The proofs (chord-margin bound, cap projection, null-space) are in the main text; keep key statements in the body and put long proofs in an appendix after references.
10. Add reproducibility/artifact statement and supplementary anonymised code archive.
11. Remove or justify unused `hendrycks2021measuring`.

## 4. Writing and storytelling review

Current state: 62 `\textbf` uses, abstract of about 330 words with three bolded method names and nine numerical results; Introduction jumps from alignment to attacks to the method name before the reader has seen one example.

Structural suggestions:
1. Open with one concrete motivating example: the same harmful request ("how to wipe logs and bash history") in plain form (refused) and wrapped in a red-team playbook persona (complied). Show it as Figure 1 with the model's responses, then show the geometric picture in Figure 2. Currently the example appears only in Table 2 near the end.
2. Frame the problem in two sentences before any notation: refusal depends on the wrapper, and defences that push activations away from jailbreaks also push away benign dual-use requests (over-refusal). State the gap: no steering defence is designed to be invariant to framing.
3. Explain "why geometry" in plain language: intent and framing are different things that the model mixes; if we can separate them, we can ask whether the intent part sits inside a harm region regardless of the framing part. Only then introduce spheres; justify spheres (direction carries meaning, norm carries scale) with one sentence citing hyperspherical representation work (Wang and Isola; Davidson et al.).
4. State the key insight as one sentence ("steer along directions the context cannot see") and a single "what changes" paragraph: where the hook sits, what is added, what is untouched for benign inputs.
5. Contributions list of three bullets with claims phrased as measured results, not bold method acronyms.
6. Related Work: reorder from broad to narrow ending with "how ARETE differs"; currently three short paragraphs miss the competitors in Section 2. Explain why prior single-direction methods (Arditi; CAST) are not invariant to context.
7. Theory section: keep one theorem and its intuition in the body; move proofs to an appendix. Add a "what the bound does and does not say" paragraph (it holds only if the encoder phi_I is actually invariant, which is measured empirically at 0.16-0.24 rad).
8. Experiments: lead each subsection with the question it answers (Does it block unseen attacks? Does it over-refuse? Does it survive adaptive attack? What does each component contribute?). Report main numbers in one table; put the long per-category results in the appendix.
9. Cut bold in running text; reserve it for table best values. Break up jargon stacks such as "Bundle-Orthogonal Intent-Context Decomposition", "Subspace Orthogonal Null-Space Steering" (name once, define once, then use "the decomposition" and "null-space steering").
10. Add a short "Failure cases" paragraph with a real example where ARETE fails or over-refuses.

Overclaiming to soften (with line numbers in main.tex):
- Line 47 (abstract): "performs exact single-step spherical-cap retraction" -> "retracts the intent vector onto a spherical-cap safe region in one step"; "preserving exact benign generation fluency" -> "leaving benign outputs unchanged on our test set".
- Line 47: "0.0% benign false-positive refusals" -> report the denominator and CI (zero events in n gives an upper bound, e.g. 95% CI [0, 3/n]).
- Line 98: "exact chord-margin safety bound" -> "closed-form chord-margin bound under assumed encoder invariance".
- Line 100: "guarantees zero linear perturbation to the context subspace" -> true only for the linear step, before the nonlinear downstream layers; say "no first-order component along the context subspace".
- Line 165 / 159 heading: "exact Riemannian metric projection", "Exact Spherical-Cap Retraction" -> "closed-form".
- Line 202: "zero downstream perturbation across all subsequent layers" -> holds only for inputs already inside the safe region, where Delta h = 0; state that.
- Line 242: "strictly inside K_safe" is a claim about the model, not the network; qualify.
- Lines 286, 294, 353: "100.0% single-step post-steering safety rate", "100.0% Harmful TPR (83/83)" -> report CIs; 83 samples cannot support "100%".
- Line 353: "preserving exact token-for-token benign outputs" -> state that the hook is inactive when m < 0, an assumption the reader can check, and give the fraction of benign inputs where it truly was inactive.
- Line 357 (conclusion): "enforces ... via exact single-step geodesic retraction" -> soften as above.
- The word "eliminating"/"guarantee"-style claims should not appear in the abstract; 2.82% under 10-step PGD is not a robustness certificate (Nasr et al. 2025 show such defences fail under stronger adaptive attacks); add an adaptive attack that targets the null-space projection and the barrier jointly, and report it.

Methodological flags a reviewer will raise (brief): evaluation only on Qwen2.5-1.5B; PGD on the residual stream (embedding-space attack) is not a realistic token-level threat model; no comparison against circuit breakers, RepBend, ReFAT, or LAT; "OOD attack families" were presumably generated by the authors; no human or LLM-judge (StrongREJECT) validation of ASR.
