# Analysis of 2503.17682v2.pdf

Total Pages: 29

## Page 1

Safe RLHF-V: Safe Reinforcement Learning
from Multi-modal Human Feedback
Jiaming Ji⋄, Xinyu Chen⋄, Rui Pan⋄, Conghui Zhang⋄, Han Zhu♯, Jiahao Li⋄
Donghai Hong⋄, Boyuan Chen⋄, Jiayi Zhou⋄, Kaile Wang⋄, Juntao Dai⋄
Chi-Min Chan♯, Yida Tang⋄, Sirui Han♯, Yike Guo♯, Yaodong Yang⋄
⋄Peking University, ♯Hong Kong University of Science and Technology
Abstract
Multimodal large language models (MLLMs) are essential for building general-
purpose AI assistants; however, they pose increasing safety risks. How can we
ensure safety alignment of MLLMs to prevent undesired behaviors? Going further,
it is critical to explore how to fine-tune MLLMs to preserve capabilities while
meeting safety constraints. Fundamentally, this challenge can be formulated as a
min-max optimization problem. However, existing datasets have not yet disentan-
gled single preference signals into explicit safety constraints, hindering systematic
investigation in this direction. Moreover, it remains an open question whether such
constraints can be effectively incorporated into the optimization process for multi-
modal models. In this work, we present the first exploration of the Safe RLHF-V
– the first multimodal safety alignment framework. The framework consists of:
(I) BeaverTails-V, the first open-source dataset featuring dual preference an-
notations for helpfulness and safety, supplemented with multi-level safety labels
(minor, moderate, severe); (II) Beaver-Guard-V, a multi-level guardrail system
to proactively defend against unsafe queries and adversarial attacks. Applying the
guard model over five rounds of filtering and regeneration significantly enhances
the precursor model’s overall safety by an average of 40.9%. (III) Based on dual
preference, we initiate the first exploration of multi-modal safety alignment
within a constrained optimization. Experimental results demonstrate that Safe
RLHF effectively improves both model helpfulness and safety. Specifically, Safe
RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.1
1
Introduction
Multimodal large language models (MLLMs) are pivotal for developing general-purpose AI assistants
[1, 2]. By using visual instruction tuning on foundation models [3], MLLMs can effectively handle
complex multimodal tasks. However, recent research highlights that images can implicitly induce
MLLMs to generate harmful content, exposing vulnerabilities that are less pronounced in purely
LLMs. What is worse, fine-tuning MLLMs can lead to the forgetting of safety alignment previously
learned by the backbone LLMs [4]. Consequently, improving the safety of MLLMs to align with
human preference is a major concern at present [5]. So, as MLLMs continue to advance, emerging
studies reveal a critical challenge:
How can we ensure the safety alignment of MLLMs? This is under exploration.
So, we need to answer: Which type of preference data and what optimization are effective?
1All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the
safety development of MLLMs and reduce potential societal risks.
Preprint. Under review.
arXiv:2503.17682v2  [cs.LG]  22 May 2025


---

## Page 2

Reinforcement Learning from Human Feedback (RLHF) has proven effective in aligning LLMs with
human preferences [6, 7, 8, 9, 10], enhancing capabilities in instruction following, reasoning, etc.
However, balancing helpfulness and safety remains challenging. Specifically, during the preference
annotation stage, it is challenging to effectively account for two conflicting metrics within a single
partial order. Helpfulness and safety are inherently somewhat contradictory; for instance, simply
refusing to answer may ensure safety but significantly reduce helpfulness. Recent works have applied
the RLHF for MLLMs to mitigate hallucinations in image understanding and mitigate trustworthiness.
RLHF-V [11] introduced human preference through segment-level corrections for hallucination
to enhance trustworthiness based on DDPO, a variant of DPO [7]. Similarly, Llava-RLHF [12]
incorporated additional factual information into the reward model to reduce reward hacking and
improve performance. However, these methods primarily focus on single-dimensional alignment for
helpfulness without explicitly addressing safety concerns in MLLMs.
Prior studies have shown that attempts to enhance the helpfulness of MLLMs can inadvertently
compromise their safety [13, 14], without guaranteeing safety. Consequently, developing methods
that enhance helpfulness without compromising safety remains an open challenge. Inspired by Safe
RLHF [15, 16], which decouples human preferences during data annotation and jointly optimizes
the dual objectives of helpfulness and safety. However, directly applying Safe RLHF to multimodal
learning proves insufficient in practice. We identify several key challenges, as detailed below:
• Lack of high-quality multimodal safety data. Existing multimodal safety preference datasets
often exhibit weak correlations between visual and textual modalities. In particular, the presence
of harmful or benign images tends to exert minimal influence on the prompt’s harmfulness.
Overall, the images and text in current datasets appear largely independent, which limits their
utility in learning grounded safety preferences.
• Direct safety meta-label annotation is infeasible. In language-only settings, explicitly cate-
gorizing safety-related preferences has proven highly effective. However, the introduction of
multimodal inputs introduces substantial inconsistencies in labeling. Annotators tend to focus on
different aspects of an image, resulting in divergent interpretations and making it challenging to
generate consistent and reliable safety preference labels.
• Whether constrained optimization remains effective in the multi-modal setting remains
to be verified. Although constrained optimization has been well-studied for language models,
its effectiveness in the multi-modal context remains uncertain and poses significant engineering
challenges. In our exploration, we found the training process in multi-modal highly unstable,
with significant difficulty in diagnosing the root cause of failures. When model performance
stagnates, it is often unclear whether the issue arises from flawed preference data, suboptimal
training hyperparameters, or - perhaps most critically - the inherent ineffectiveness of constrained
optimization in multi-modal settings.
In this work, we propose Safe RLHF-V, a novel framework designed to safety alignment of MLLMs,
addressing the aforementioned challenges through innovations in data construction, annotation
strategy, and optimization methodology. Our main contributions are as follows:
• (Dataset) We have open-sourced BeaverTails-V, the first dataset featuring dual preference
annotations for both helpfulness and safety in MLLMs. For each pair, we independently annotated
preferences regarding helpfulness and safety. Additionally, we provided graded safety labels:
minor, moderate, and severe to reduce the inconsistencies in labeling, establishing the first
exploration to enable multi-level safety alignment in MLLMs.
• (Guardrail) Utilizing the multi-level safety meta labels, we developed a multi-level guardrail
system to filter red-teaming and unsafe queries within MLLM systems. Experimental results
demonstrate that Beaver-Guard-V outperforms existing methods across multiple safety categories
and effectively strengthens the model’s defense against varying levels of safety risks through
graded safety management.
• (Algorithm) We introduce Safe RLHF-V, the first multimodal safety alignment algorithm that
balances helpfulness and safety. This balance is achieved through a Lagrangian-based min-
max optimization framework. To reduce the unstable optimization, we introduce the budget
bound. Experimental results show that Safe RLHF-V significantly enhances model safety without
sacrificing helpfulness. Specifically, Safe RLHF-V, the model’s safety improved by 34.2%, while
helpfulness increased by 34.3%.
2


---

## Page 3

All datasets, models, and code have been open-sourced. We hope this work can facilitate the safety
alignment of MLLMs, thereby mitigating their potential societal risks.
2
Related Work
Alignment and Safety Concerns in MLLMs.
Recent LLM [17, 18, 19, 20, 21] and MLLMs
[22, 23, 2, 24, 24, 25] have raised safety concerns, including generate offensive content [21, 26], leak
personal information [27, 28], and propagate misinformation [29]. Alignment aims to ensure these
AI models are following human intention and values [5]. In practice, the widely accepted standard
for large models is the 3H principle: Helpful, Honest, and Harmless [30]. Reinforcement Learning
from Human Feedback (RLHF) [6] trains a reward model (RM) on a human preference dataset and
uses the reward signal provided by the RM to fine-tune via reinforcement learning [31]. It is now
widely applied to align LLMs and its variants [7]. Inspired by this, many methods that learn from
human or AI preferences have been developed for aligning MLLMs [12, 11, 32]. LLaVA-RLHF [12]
trains VLLMs and augments the reward model with additional factual information, mitigating reward
hacking in RLHF. RLHF-V [11] further enhances MLLM trustworthiness through behavior alignment
based on fine-grained corrective human feedback. Although these approaches have significantly
improved the performance of MLLMs, they do not explicitly guarantee model safety, particularly
when handling toxicity-related queries.
Safety Preference Data and Evaluation. Evaluating the safety of MLLMs has become a critical
area of research [21, 33, 34, 35], especially as these models are increasingly deployed in real-world
applications. Various benchmarks have been developed to assess the robustness of MLLMs against
potential vulnerabilities. SPA-VL [36] focuses on how models adapt to safety preferences, but its
images mainly depict safe scenarios, and its image-text pairs are not well decoupled, limiting the
model’s ability to fully leverage multimodal interactions. Unsafebench [37] evaluates task executions
in hazardous scenarios, but lacks sufficient diversity in dangerous situations. MM-SafetyBench
[38] assesses models’ understanding of safety preferences, but is more focused on evaluation than
providing comprehensive training data. Although these datasets provide valuable information on
multimodal safety preferences, they face limitations in balancing helpfulness and safety. To address
these shortcomings, we propose Beavertails-V, aiming to offer a more diverse and comprehensive
dataset for multimodal safety preferences.
3
Datasets
3.1
Why build a new safety MLLMs dataset?
Table 1: Comparison of various multimodal
harmlessness datasets with attack success rate.
Performance
Datasets
Models
Text-only
Text-Image
Delta↑
SPA-VL
LLaVA-1.5-7B
0.213
0.401
0.188
Qwen2-VL-7B
0.040
0.108
0.068
MM-
LLaVA-1.5-7B
0.041
0.233
0.192
Safetybench
Qwen2-VL-7B
0.002
0.121
0.119
Beavertails-V
LLaVA-1.5-7B
0.349
0.565
0.216
(Ours)
Qwen2-VL-7B
0.036
0.276
0.240
Reasoning #1
The current open-source multi-
modal harmlessness datasets exhibit low consis-
tency between text and image prompt inference.
To investigate this, we conducted a correlation
evaluation across BeaverTails-V, SPA-VL [36],
and MM-SafetyBench [39], with results summa-
rized in Table 1. For each harmful inference,
two types of input were tested: (1) the original
text-image pair and (2) the text-only input. We
utilized the general-purpose MLLM – LLaVA-
1.5-7B [3] and Qwen2-VL-7B [25] to perform adversarial attacks, measuring the attack success rate
(ASR) [40, 39]. ASR is defined as the proportion of unsafe responses generated, and its calculation
involves using the GPT-4o [13] API to assess the safety of the generated responses. Experimental
results reveal that image content minimally influences the harmfulness of queries in existing datasets.
For example, in SPA-VL evaluated with Qwen2-VL-7B, the ASR with text-only input was 4.0%,
increasing marginally to 10.8% with the original text-image pair – an increment of only 6.8%. This
outcome reinforces our hypothesis that current multimodal safety alignment datasets lack genuine
multimodal integration.
Reasoning #2 . Compared to existing datasets, BeaverTails-V offers unignorable advantages in
the diversity of harm categories and well-refined annotations, as shown in Fig. 1. BeaverTails-V
encompasses more harm categories with a balanced distribution between safe and unsafe data. For
3


---

## Page 4

0
500
1000
1500
2000
2500
3000
3500
Terrorism or Extremism
Hacking or Digital Crime
Harmful Health Content
Environmental Damage
Dangerous Behavior
Violence and Physical Harm
Psychological Harm and Manipulation
Animal Abuse
Privacy Invasion and Surveillance
Identity Misuse and Impersonation
Discriminatory Depictions
Insulting and Harassing Behavior
Deception in Personal Relationships
Financial and Academic Fraud
False Information
Sensitive Information in Key Areas
Horror and Gore
Psychological Horror and Dark Themes
Pornographic Content
Sexual Crimes
Safe
Minor
Moderate
Severe
Data distribution of BeaverTails-V (ours)
0
400
800
1200
1600
2000
Persuasion and manipulation
Violation of personal property
Dangerous information
Privacy
Defamation
Fraud or deceptive action
Illegal activities
Influence operations
Security threats
Erosion of trust in public…
False beliefs
Toxic
Unfair
Risky financial practices
Trade and compliance
0
200
400
600
800
Deception
Harassment
Hate
Illegal activity
Political
Public and personal…
Self-harm
Sexual
Shocking
Spam
Violence
Safe
Unsafe
UnSafeBench
SPA-VL
Figure 1: Safety annotation distribution of various multimodal safety dataset. We labeled the type
of images as safe and unsafe (including minor, moderate, and severe) by expert annotations. We find
that existing datasets significantly overestimate the proportion of safe images. In contrast, BeaverTails-
V presents a lower proportion of safe images, more accurately reflecting harmful categories.
example, over 50% images in categories such as unfair behavior and high-risk financial activities are
labeled as safe in SPA-VL. This imbalance limits the applicability in real-world scenarios. Further-
more, BeaverTails-V achieves an orthogonal decoupling of helpfulness and safety while incorporating
multi-level safety grading annotations to facilitate hierarchical safety alignment. Specifically, beyond
categorizing QA pairs as safe or unsafe, we further assign meta-labels indicating the severity of harm
(i.e. minor, moderate, and severe) to unsafe QA pairs.
3.2
Data Collection and Annotation Process:
To construct the high-quality BeaverTails-V dataset, we designed a multi-stage data collection and
annotation process as described below. A detailed pipeline document can be referred to Appendix B.
,→Harmful Categories. Drawing inspiration from the text-based safety datasets [41, 15], and
guided by usage policies and human assessment, we constructed a multimodal safety taxonomy
with 9 primary and 20 secondary categories, as shown in Fig. 5. Detailed descriptions of each
category are provided in Appendix B.2.
,→Prompt and Image Generation: We constructed our dataset by first retrieving diverse, harm-
aligned images from Yandex, guided by a taxonomy-driven set of GPT-4 generated keywords
that were manually refined to ensure category fidelity. To balance harm-level representation, we
dynamically adapted search strategies and performed multi-round manual filtering for quality
control. For each image, we then used the Ovis1.6-Gemma2-9B model to generate eight diverse
and contextually rich prompts via a combination of instruction-following and jailbreak strategies,
aiming to capture the nuanced semantics of harmful content.
,→Response Generation: We utilized various VLLMs to generate responses for 32k prompts.
To maximize diversity in response pairs, we applied a two-step filtering strategy: (1) one
response was randomly selected from five model outputs (excluding Phi-3.5-vision-instruct
[42]) as response a. (2) using paraphrase-MiniLM-L6-v2 [43] embeddings and cosine similarity,
we selected the most dissimilar response as response b. Existing models often struggled to
generate concise rejection responses, leading to out-of-distribution issues. To mitigate this, we
incorporated rejection responses generated by the Phi-3.5-vision-instruct model at a 4% ratio.
This embedding-based selection method effectively enhanced dataset diversity.
,→Question-Answer Pair Annotation: Prior works have shown that GPT-4o matches human
annotators’ accuracy and preference consistency for data annotation [44]. Inspired by PKU-
SafeRLHF [45], we identified distinct gradations in the helpfulness and safety of model responses.
While models can generate basic rejections, these are generally less safe than responses offering
proactive warnings and detailed explanations. We defined safety on a 7-point scale, with a score
4


---

## Page 5

Beaver-Guard-V API {Prompt, Response}
Abuse
…
Sex
Harmless
What toxic methods could be used to clean the area depicted 
in the image?
Prompt
Response Generation & Sampling
Harmless?
Helpful?
Yes
Unhelpful
Helpful
No
No
Reject & Regenerate
Reject & Regenerate
Yes
Moderation
Chemical cleaning agents could be used to break down the waste, but this 
would likely be harmful to the environment. What other methods could be 
cleaned the area?
Response
Harmless & Helpful
Violence
Prompt
Moderation API {Prompt}
Does the prompt violate usage policies?
No
Yes
Prompt Filtering
Text-Image Analysis
Refuse to Answer
Generation & Sampling
Moderation
(a) Moderation
(b) Our Moderation: Beaver-Guard-V
Response
As an AI multimodal model, 
I am unable to engage with 
content that may violate my 
usage guidelines.
Toxic methods to clean 
such an area might include: 
1. Chemical, 2. Heavy 
Metals, …
Chemical cleaning … 
would likely be harmful to 
the environment. 
Response
Response
Unhelpful
Harmless & Helpful
Harmful
What toxic methods could be used to clean the area depicted 
in the image?
(c) Moderation for Determining QA-Pair Level
@0
@1
@3
@5
LlaVA-1.5-7B
Qwen2-VL-7B
LLaVA-v1.6-13B
0.502
0.326
0.343
0.398
0.275
0.220
0.316
0.258
0.173
0.154
0.249
0.281
(d) Attack Success Rate under Different Moderation Rounds
Beaver-Guard-V API
What toxic methods could be used to clean the area 
depicted in the image?
Prompt
Minor
Moderate
Severe
Safe
QA-Pair Level: Severe.
QA-pair level?
Figure 2: Illustration of Beaver-Guard-V’s Moderation Pipeline. (a)&(b): Traditional moderation
filters toxic prompts pre-generation, often refusing to respond, which can yield unhelpful or harmful
outputs. Beaver-Guard-V adopts a two-stage approach, moderating both prompts and responses post-
generation via rejection and regeneration to ensure safe, helpful outputs. (c): It provides fine-grained
QA-pair annotations (Safe, Minor, Moderate, Severe) for downstream safety analysis. (d): Across
multiple moderation rounds, Beaver-Guard-V significantly lowers attack success rates on models.
of 0 representing a neutral balance between safety and harm. Negative scores (-1 to -3) denote
increasing levels of harmfulness, whereas non-negative scores indicate ascending degrees of
safety. In particular, scores of 0 and 1 correspond to basic rejections without guidance, while
scores of 2 and 3 reflect stronger advisory tones and responsibilities.
Table 2: Comparison of Beaver-Guard-V with baselines
Table 2a: Comparison between Beaver-Guard-V and other methods.
Models
Metrics
Accuracy
Precision
Recall
F1-Score↑
False Positive Rate↓
Llama-Guard-3-11B-Vision
Safety
0.64
0.67
0.84
0.74
0.71
Beaver-Guard-V-Binary (Ours)
Safety
0.78
0.77
0.93
0.84
0.47
Beaver-Guard-V-Multi-Level (Ours)
Safety
0.85
0.88
0.87
0.88
0.20
Table 2b: Comparison between Beaver-Guard-V and Llama-Guard-3-11B-Vision.
BeaverTails-V
SPA-VL
MM-SafetyBench
Models
Metrics
Eval
Test
SD
SD_TYPO
TYPO
Average
Llama-Guard-3-11B-Vision
Accuracy
0.64
0.64
0.70
0.65
0.68
0.68
Beaver-Guard-V (Ours)
Accuracy
0.85
0.88
0.88
0.85
0.90
0.88
Llama-Guard-3-11B-Vision
F1-Score
0.74
0.74
0.80
0.74
0.79
0.78
Beaver-Guard-V (Ours)
F1-Score
0.88
0.92
0.93
0.90
0.94
0.92
3.3
Moderation Application: Beaver-Guard-V
Due to the inherent vulnerability of MLLMs, the guardrail model filters out harmful user queries and
model responses to ensure safety generation [46]. We utilize safety taxonomy labels (i.e., safe or
unsafe) and multi-level safety meta tags (i.e., minor, moderate, and severe) from the BeaverTails-V
to fine-tune the Beaver-Guard-V model for detecting diverse forms of harmful content (see Fig. 2).
Experiment results show that the Beaver-Guard-V model accurately classifies specific harmful content
categories, enabling targeted filtering strategies tailored to different content types.
To evaluate its effectiveness in identifying toxic content, we assess the Beaver-Guard-V model on
both binary and multi-level meta label settings. As shown in Table 2, the model achieves a high
accuracy rate of 78% in binary setting. Notably, it maintains a low false positive rate of 47%,
effectively identifying harmful content while minimizing the erroneous flagging of benign responses.
This balance is particularly crucial in real-world applications, where false positives can disrupt user
experience and compromise model reliability.
In the multi-level meta label setting, the model also demonstrates strong performance. Supported by
the well-annotated Bevertails-V comprising 20 distinct harmful content categories and multi-level
5


---

## Page 6

meta label of QA pairs, the model attains an 85% accuracy rate in detecting harmful content. Notably,
Beaver-Guard-V attains state-of-the-art results on several benchmarks. Furthermore, Beaver-Guard-V
can also classify both the category of harmful content and its level of severity. This fine-grained
classification capability allows Beaver-Guard-V to manage a broad spectrum of harmful content with
high precision. As shown in Fig. 2, we further evaluate Beaver-Guard-V’s effectiveness in reducing
attack success rates (ASR) under different detection rounds (aka, Filter of N; FoN). The results reveal
that increasing moderation rounds consistently lower ASR across different MLLMs. Specifically, at a
moderation round of 5, Beaver-Guard-V achieves the lowest ASR across all evaluated models. This
model can be used to emphasize the importance of adaptive moderation strategies in maintaining the
safety of MLLM interactions.
4
Method: Safe RLHF-V
4.1
Background and Preliminaries
Supervised Fine-tuning.
RLHF begins with a pre-trained model, which is then fine-tuned via
supervised learning on a high-quality human instruction dataset designed for downstream tasks. This
process results in an chat model θSFT.
Reward Modeling. Then, we use preference data D to train the RM rRM(·|·). The chat model θSFT
generates response pairs (y1, y2) from given prompts x. Human annotators are then tasked with
selecting their preferred response from each pair, denoted as yw ≻yl | x, where yw and yl denote
the preferred and preferred answer amongst (y1, y2). So, the RM rRM(·|·) can be trained by,
L = E§∼D[log σ(rθ(x, yl) −rθ(x, yw))].
(1)
RL Fine-tuning. Finally, we optimize the LLM via RL, guided by the reward model rRM, where a
reward is obtained from rRM at the end of each response. The primary objective of RL is to adjust
the LLM parameters θLM to maximize the expected reward on the prompt distribution P. That is,
θLM = arg max
θ
Ex∼P,y∼θLM [rRM (y | x)] .
(2)
Safe reinforcement learning aims to ensure that policy learning adheres to safety constraints typically
formulated within the CMDP framework [47, 48].
Constrained Markov Decision Process is defined by the tuple (S, A, R, P, µ, γ, C). Here S is state
space, A is action space, P (s′ | s, a) is probability of state transition from s to s′ after playing a.
r(·) : S ×A×S →R and r (s′ | s, a) denote the reward that the agent observes when state transition
from s to s′ after it plays a. µ(·) : S →[0, 1] is the initial state distribution and γ ∈(0, 1). A
stationary parameterized policy πθ is a probability distribution defined on S × A, πθ(a | s) denotes
the probability of playing a in state s. The goal of RL is to maximize the,
J (πθ) = Es∼µ
"
Eπθ
" ∞
X
t=0
γtrt+1 | s0 = s
##
.
The set C = {(ci, bi)}m
i=1, where ci are cost functions: ci : S × A →R, and cost thresholds are
bi, i = 1, . . . , m. So, the objectives of SafeRL as follows,
π⋆= arg max
πθ∈ΠC
J (πθ) , s.t. Eπθ
" ∞
X
t=0
γtci (st, at)
#
≤bi.
(3)
4.2
Multimodal Reward and Cost Model
Inspired by [16], we train two independent preference models to fit human preference across the
helpfulness and safety of MLLM responses. The reward model is developed from the helpfulness
dataset DR, providing the reward signals optimized for helpfulness during the RL phase. The cost
model is built upon the safety dataset DC, delivering insights into human perceptions regarding the
safety of LLM responses.
6


---

## Page 7

Reward Model-Vision (RM-V). Utilizing the helpfulness dataset DR =
 xi, yi
w, yi
l
	N
i=1, we
train a parameterized RM Rφ(y, x), where Rφ(y, x) represents a scalar output. This model is trained
with the pairwise comparison loss,
LR = −Ee∼DR [log σ (Rφ(yw, x) −Rφ(yl, x))] .
(4)
Cost Model Vision (CM-V). Unlike the helpfulness human preference dataset, the harmlessness
human preference dataset provides additional information about the harmlessness of a response. To
make optimal use of this information for training the CM Cφ(y, x), we amend the original pairwise
comparison loss by incorporating classification terms, similar to Safe RLHF [16],
LC = −Ee∼DC [log σ (Cψ(yw, x) −Cψ(yl, x))]
−k ∗E(x,yw,yl,sw,sl)∼DC [log σ (sw · Cψ(yw, x)) + log σ (sl · Cψ(yl, x))] ,
(5)
where k scales the classification loss of harmful and safe responses, allowing control over its relative
importance in the overall cost model optimization.
4.3
Reinforcement Learning from Multimodal Human Feedback
During the RL phase, we utilize the Reward Model Rϕ to estimate the value of human preference for
helpfulness, while the Cost Model Cψ for safety. The following optimization objective is a Safe RL
scheme previously outlined in, thereby defined as the objective for our Safe RLHF-V setting:
max
θ
Ex∼D,y∼πθ(·|x) [Rϕ(y, x)] , s.t. Cψ(y, x) ≤0,
∀x ∼D, y ∼πθ(·|x),
(6)
where D is a distribution of prompts used in the RL phase, and the y = a1:T are responses generated
by the MLLM πθ. This equation encapsulates our primary goal: to maximize the expected reward
within the constraints of ensuring the harmlessness of the responses generated by the MLLMs.
Min-Max Optimization with Budget Bound
To address this constrained problem, we leverage
the Lagrangian method to find the local maxima and minima of a function over a constraint set and
its unconstrained Lagrangian dual form as follows:
min
θ
max
λ≥0 [−JR(θ) + λ · JC(θ)].
(7)
Unlike in language-only settings, we find that the values of the Lagrange multipliers λ significantly
affect optimization stability in multimodal scenarios. In practice, naive Lagrange updates can lead to
model collapse. To stabilize learning process, we introduce a Budget Bound in the update as follows:
λ ←proj
λ
[λ −α(b −JC(θ))],
(8)
where α is the step size. The projection operator projλ projects λ back into the interval [0, νmax]
where νmax is chosen so that λ does not become too large. In practice, optimizing helpfulness JR
often conflicts with minimizing harm JC. Eq. 7 introduces a penalty term, modulated by λ, to balance
these objectives. By iteratively solving the min-max problem, updating both model parameters and λ,
the framework dynamically adjusts to changes in potential harm, preventing imbalanced optimization.
5
Experiments
In this section, we seek to address the following questions:
• Q1: How does Safe RLHF-V compare to existing multimodal alignment methods?
• Q2: How robust are reward and cost models in Safe RLHF-V with respect to preference data?
• Q3: Lagrange multipliers regulate the trade-off in the min-max optimization. Does the algorithm
perform well for different values of multipliers?
7


---

## Page 8

Table 3: Comparison of Safe RLHF-V and baselines. Across various benchmarks, Safe RLHF-V
provides the strongest performance in balancing helpfulness and safety, significantly outperforming
RLHF. We train the single preference algorithm using only single-dimensional preferences. The
evaluation of the method is grounded in safety alignment application scenarios. We first consider the
safety improvement, followed by the overall enhancement across the two dimensions.
Beavertails-V
MM-SafetyBench
SPA-VL
VLGuard
VLSBench
Models
Opt. Types
Safety↑
Helpful↑
Safety↑
Helpful↑
Safety↑
Helpful↑
Safety↑
Helpful↑
Safety↑
Helpful↑
Llava-1.5-7B
N/A
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
+RLHF-V
Single
0.562 + 0.062 0.110 - 0.390
0.687 + 0.187 0.174 - 0.326
0.341 - 0.159 0.207
#
0.462 - 0.038 0.410
#
0.311 - 0.189 0.142
#
+Llava-RLHF
Single
0.435 - 0.065 0.738
#
0.514 + 0.014 0.665 + 0.165 0.446 - 0.054 0.591
#
0.487 - 0.013 0.413
#
0.548 + 0.048 0.645 + 0.145
+DPO (Helpful*)
Single
0.235 - 0.265 0.851
#
0.529 + 0.029 0.697 + 0.197 0.491 - 0.009 0.692
#
0.486 - 0.014 0.667
#
0.441 - 0.059 0.814
#
+DPO (Safety*)
Single
0.793 + 0.293 0.441 - 0.059
0.766 + 0.266 0.459 - 0.041
0.762 + 0.262 0.339 - 0.161
0.910 + 0.410 0.239 - 0.261
0.667 + 0.167 0.482 - 0.018
+PPO (Helpful)
Single
0.528 + 0.028 0.672 + 0.172 0.649 + 0.149 0.563 + 0.063 0.593 + 0.093 0.547 + 0.047 0.507 + 0.007 0.562 + 0.062 0.495 - 0.005 0.672
#
+PPO (Safety)
Single
0.957 + 0.457 0.253 - 0.247
0.717 + 0.217 0.524 + 0.024 0.947 + 0.447 0.433 - 0.067
0.731 + 0.231 0.585 + 0.085 0.740 + 0.240 0.453 - 0.047
+MM-RLHF (Helpful)
Single
0.382 - 0.118 0.534
#
0.575 + 0.075 0.660 + 0.160 0.438 - 0.062 0.716
#
0.555 + 0.055 0.715 + 0.215 0.372 - 0.128 0.552
#
+MM-RLHF (Safety)
Single
0.837 + 0.337 0.345 - 0.155
0.845 + 0.345 0.421 - 0.079
0.703 + 0.203 0.352 - 0.148
0.696 + 0.196 0.369 - 0.131
0.619 + 0.119 0.348 - 0.152
+SafeRLHF-V
Dual
0.579 + 0.079 0.770 + 0.270 0.720 + 0.220 0.774 + 0.274 0.620 + 0.120 0.613 + 0.113 0.852 + 0.352 0.793 + 0.293 0.611 + 0.111 0.814 + 0.314
Qwen2-VL-7B
N/A
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
+RLHF-V*
Single
0.452 - 0.048 0.544
#
0.491 - 0.009 0.447
#
0.322 - 0.178 0.370
#
0.477 - 0.023 0.773
#
0.391 - 0.109 0.761
#
+Llava-RLHF
Single
0.549 + 0.049 0.647 + 0.147 0.521 + 0.021 0.504 + 0.004 0.429 - 0.071 0.441
#
0.541 + 0.041 0.748 + 0.248 0.455 - 0.045 0.807
#
+DPO (Helpful)
Single
0.630 + 0.130 0.796 + 0.296 0.507 + 0.007 0.595 + 0.095 0.557 + 0.057 0.585 + 0.085 0.377 - 0.123 0.623
#
0.475 - 0.025 0.859
#
+DPO (Safety)
Single
0.831 + 0.331 0.599 + 0.099 0.799 + 0.299 0.536 + 0.036 0.634 + 0.134 0.599 + 0.099 0.596 + 0.096 0.625 + 0.125 0.530 + 0.030 0.747 + 0.247
+PPO (Helpful)
Single
0.506 + 0.006 0.651 + 0.151 0.477 - 0.023 0.533
#
0.602 + 0.102 0.566 + 0.066 0.602 + 0.102 0.612 + 0.112 0.437 - 0.063 0.859
#
+PPO (Safety)
Single
0.883 + 0.383 0.357 - 0.143
0.761 + 0.261 0.384 - 0.116
0.713 + 0.213 0.456 - 0.044
0.876 + 0.376 0.620 + 0.120 0.523 + 0.023 0.523 + 0.023
+MM-RLHF (Helpful)
Single
0.458 - 0.042 0.462
#
0.418 - 0.082 0.522
#
0.512 + 0.012 0.766 + 0.266 0.380 - 0.120 0.888
#
0.384 - 0.116 0.826
#
+MM-RLHF (Safety)
Single
0.722 + 0.222 0.443 - 0.057
0.765 + 0.265 0.217 - 0.283
0.787 + 0.287 0.443 - 0.057
0.832 + 0.332 0.575 + 0.075 0.685 + 0.185 0.436 - 0.064
+SafeRLHF-V
Dual
0.689 + 0.189 0.720 + 0.220 0.780 + 0.280 0.578 + 0.078 0.683 + 0.183 0.626 + 0.126 0.780 + 0.280 0.873 + 0.373 0.605 + 0.105 0.850 + 0.350
LLaVA-1.6-7B
N/A
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
0.5
+RLHF-V*
Single
0.421 - 0.079 0.065
#
0.398 - 0.102 0.154
#
0.406 - 0.094 0.250
#
0.721 + 0.221 0.603 + 0.103 0.274 - 0.226 0.139
#
+Llava-RLHF
Single
0.581 + 0.081 0.449 - 0.051
0.638 + 0.138 0.594 + 0.094 0.541 + 0.041 0.607 + 0.107 0.797 + 0.297 0.648 + 0.148 0.454 - 0.046 0.649
#
+DPO (Helpful)
Single
0.494 - 0.006 0.737
#
0.579 + 0.079 0.537 + 0.037 0.440 - 0.060 0.519
#
0.461 - 0.039 0.513
#
0.444 - 0.056 0.698
#
+DPO (Safety)
Single
0.758 + 0.258 0.379 - 0.121
0.676 + 0.176 0.431 - 0.069
0.543 + 0.043 0.418 - 0.082
0.572 + 0.072 0.503 + 0.003 0.599 + 0.099 0.494 - 0.006
+PPO (Helpful)
Single
0.451 - 0.049 0.610
#
0.631 + 0.131 0.719 + 0.219 0.561 + 0.061 0.658 + 0.158 0.409 - 0.091 0.625
#
0.308 - 0.192 0.451
#
+PPO (Safety)
Single
0.594 + 0.094 0.443 - 0.057
0.592 + 0.092 0.554 + 0.054 0.698 + 0.198 0.604 + 0.104 0.526 + 0.026 0.357 - 0.143
0.579 + 0.079 0.435 - 0.065
+MM-RLHF (Helpful)
Single
0.404 - 0.096 0.531
#
0.518 + 0.018 0.627 + 0.127 0.497 - 0.003 0.807
#
0.497 - 0.003 0.744
#
0.440 - 0.060 0.717
#
+MM-RLHF (Safety)
Single
0.838 + 0.338 0.394 - 0.106
0.775 + 0.275 0.273 - 0.227
0.874 + 0.374 0.564 + 0.064 0.721 + 0.221 0.321 - 0.179
0.722 + 0.222 0.354 - 0.146
+SafeRLHF-V
Dual
0.580 + 0.080 0.630 + 0.130 0.705 + 0.205 0.704 + 0.204 0.684 + 0.184 0.679 + 0.179 0.832 + 0.332 0.722 + 0.222 0.534 + 0.034 0.622 + 0.122
5.1
Experiment Setup
Datasets and Models.
To ensure the safety of MLLMs, we use Beavertails-V as the training
dataset. Specifically, we utilize the helpfulness and safety preference to train the RM-V and CM-
V, respectively. Our experiments are conducted on various MLLMs: Llava-7B-(1.5 and 1.6) and
Qwen2-VL-7B. Furthermore, we fine-tune the original model using RLHF [6] and DPO [7] with
single-dimension annotations and compare them against RLHF-V [11], Llava-RLHF [12], and
MM-RLHF [49] as the comparison baseline.
Evaluation Metrics. Given the absence of publicly available multimodal evaluation datasets that
simultaneously assess helpfulness and safety, we constructed our own evaluation set of BeaverTails-V.
We employ the win rate metric to quantify improvements in model performance. Model outputs are
evaluated using GPT-4o, and the specific prompts utilized for evaluation are detailed in Appendix C.
5.2
Main Results
We use win rate as the key metric, and the pairwise comparisons of model outputs (assessed via GPT-
4o) indicate overall improvement. As shown in Table 3, Safe RLHF-V delivers the best performance
in optimizing helpfulness while ensuring compliance with safety constraints. Notably, Safe RLHF-V
surpasses RLHF in safety and outperforms RLHF trained solely with helpfulness in the single dimen-
sion. For instance, in Beavertails-V, Safe RLHF-V consistently exceeds RLHF across various models
in dual dimensions. A key factor behind this improvement is CM-V, which effectively constrains the
exploration space during policy optimization, ensuring stable convergence. While single-dimensional
DPO achieves high helpfulness scores via contrastive learning, its safety performance is significantly
lower, underscoring the difficulty of balancing helpfulness and harmlessness without explicit safety
modeling. In real-world scenarios, preference data from diverse annotators reflects varying values
and backgrounds, introducing multi-scale information. As a result, binary preferences alone often
fail to ensure stable convergence, especially in safety-critical settings.
8


---

## Page 9

!"#$%&%'
!"#$%&'
!"#$'
!"#$'%
!"#$'%%
!$%&%'
!$%&'
!$'
!$'%
!$'%%
%&(
%&)
%&*
%&+
%&,
%&-
%&(
%&)
%&*
%&+
%&,
%&-
!"#$%&'($) *(+,-.+#(//
!"#$%&'($) *&01+(//#(//
.!/012342+5
Figure 3: Left: Ablation: static reward shaping and Safe RLHF-V. We find that reward shaping tends
to improve single dimension, while our method maintains a consistent improvement for dual objective.
w/o and w denote the static reward shaping coefficients and Safe RLHF-V. Right: The trends of λ,
reward, and cost reflect the constrained min-max optimization principle behind the algorithm.
5.3
Ablation Study
How much preference data do RM-V and CM-V require?
RM-V and CM-V serve as essential
optimization signals within the Safe RLHF-V pipeline. To assess model accuracy across different
data scales, we randomly selected varying proportions of preference data to train RM-V and CM-V
separately. As shown in Table 4, our results indicate that the multimodal reward model achieves 82.2%
accuracy with just 5K preference data points. In contrast, due to the complexity of safety constraints,
CM-V attains only 54.5% accuracy under the same 5K safety preference data, highlighting the
inherent challenges in modeling safety constraints. However, as the dataset size increases, CM-V’s
accuracy improves, reaching 79% at 15K data points. From a practical perspective, this data volume
remains acceptable for real-world applications.
Table 4: The accuracy of RM-V and CM-V with
different preference data sizes. We find that as
the training set size increases, the accuracy of both
RM-V and CM-V also improves. CM (sign)* de-
notes the accuracy of CM-V in correctly distin-
guishing between safe and unsafe categories.
Data →
5k
10k
15k
20k
25k
30K
RM
82.2%
83.6%
84.5%
85.4%
85.8%
86.3%
CM
54.5%
76.6%
79.3%
79.9%
79.7%
80.4%
CM (sign)*
66.1%
87.8%
87.8%
88.7%
88.8%
89.7%
Sensitive of initial value of λ0. We conducted
the comparison between dynamic update mech-
anism and reward shaping with different initial
values of the multiplier on the Qwen2-VL-7B
model. As shown in Fig. 3, when the multiplier
remains fixed, the model’s performance deteri-
orates significantly as the initial value changes.
The win rates of helpfulness and safety are
formed like a parabola. This means that the
model is very sensitive to the initial value of λ0
and can hardly meet both good helpfulness and
safety without a carefully-assigned initial value. In extreme cases, the model results in collapse
or produces illogical outputs and even gibberish. On the contrary, we find that when applying a
dynamic update mechanism (Eq. 7), the final converged model’s helpfulness and safety show slight
fluctuations at a win rate of about 70%. This demonstrates that the dynamic update mechanism is
insensitive to the initial values.
5.4
The Training Curve and Budget Bound Analysis
As shown in Fig. 3, we illustrate the evolution of λ during training and its correlation with reward and
cost, offering clearer insights into the constrained min-max optimization principle of Safe RLHF-V.
In the early training stage, the unaligned initial model exceeds the predefined safety threshold. At this
stage, λ is rapidly activated and quickly reaches the budget bound. Subsequently, λ stays at its upper
bound, strongly penalizing the generation of unsafe responses. This gradually lowers the model’s
likelihood of generating unsafe outputs. In the oscillation phase, λ follows the min-max principle,
exhibiting oscillations along with reward and cost fluctuations. During this phase, the cost remains
near the threshold as the model seeks to maximize reward under the given constraints.
9


---

## Page 10

In the final convergence phase, the cost and reward curves stabilize, and λ gradually becomes inactive,
declining from budget bound to zero. The red-shaded area shows that during fine-tuning, the model’s
probability of generating unsafe responses steadily decreases. At this stage, the model optimizes
responses while adhering to safety constraints, greatly improving reward model safety. This dynamic
adjustment enables Safe RLHF-V to enhance model performance while ensuring safety compliance.
6
Conclusion and Outlook
This work introduces Safe RLHF-V, the first comprehensive framework for multimodal safety
alignment. Central to our approach is BeaverTails-V, the first open-source dataset with dual preference
annotations and multi-level safety labels, enabling fine-grained supervision of both helpfulness and
safety. Through constrained optimization and the Beaver-Guard-V defense system, our method
achieves significant improvements in safety (+34.2%) and helpfulness (+34.3%). These results
highlight the potential of dual-supervised data and principled optimization for building safer, more
capable AI assistants. While our study focuses on image-text scenarios, broader generalization
to other modalities remains a limitation. Future work will extend this framework to incorporate
additional modalities and adaptation mechanisms to address evolving safety challenges in AI systems.
References
[1] OpenAI. Gpt-4v(ision) system card. https://openai.com/index/gpt-4v-system-card,
2023.
[2] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: En-
hancing vision-language understanding with advanced large language models. arXiv preprint
arXiv:2304.10592, 2023.
[3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances
in neural information processing systems, 36, 2024.
[4] Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. Safety
fine-tuning at (almost) no cost: A baseline for vision large language models. In Forty-first
International Conference on Machine Learning, 2024.
[5] Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan,
Zhonghao He, Jiayi Zhou, Zhaowei Zhang, et al. Ai alignment: A comprehensive survey. arXiv
preprint arXiv:2310.19852, 2023.
[6] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin,
Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to
follow instructions with human feedback. Advances in Neural Information Processing Systems,
35:27730–27744, 2022.
[7] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model.
In Thirty-seventh Conference on Neural Information Processing Systems, 2023.
[8] Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase,
Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, et al. Foundational
challenges in assuring alignment and safety of large language models.
arXiv preprint
arXiv:2404.09932, 2024.
[9] Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, and Yu Qiao.
Emulated disalignment: Safety alignment for large language models may backfire!
arXiv
preprint arXiv:2402.12343, 2024.
[10] Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Tekin, and Ling Liu. Lisa: Lazy safety
alignment for large language models against harmful fine-tuning attack. Advances in Neural
Information Processing Systems, 37:104521–104555, 2024.
10


---

## Page 11

[11] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan
Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior
alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 13807–13816, 2024.
[12] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang
Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models
with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023.
[13] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, et al. Gpt-4
technical report. arXiv preprint arXiv:2303.08774, 2023.
[14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783, 2024.
[15] Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang
Sun, et al. Beavertails: Towards improved safety alignment of llm via a human-preference
dataset. Advances in Neural Information Processing Systems, 36, 2024.
[16] Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and
Yaodong Yang. Safe rlhf: Safe reinforcement learning from human feedback. In The Twelfth
International Conference on Learning Representations, 2024.
[17] Somnath Banerjee, Sayan Layek, Soham Tripathy, Shanu Kumar, Animesh Mukherjee, and
Rima Hazra. Safeinfer: Context adaptive decoding time safety alignment for large language
models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages
27188–27196, 2025.
[18] Anthony Mahene, Daniel Pereira, Vincent Kowalski, Elizabeth Novak, Catherine Moretti, and
Josephine Laurent. Automated dynamic data generation for safety alignment in large language
models. Authorea Preprints, 2024.
[19] Rishabh Bhardwaj and Soujanya Poria. Red-teaming large language models using chain of
utterances for safety-alignment. arXiv preprint arXiv:2308.09662, 2023.
[20] Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson.
Fine-tuning aligned language models compromises safety, even when users do not intend to!
arXiv preprint arXiv:2310.03693, 2023.
[21] Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, and Yu Qiao. Safety of multimodal large
language models on images and text. In Proceedings of the Thirty-Third International Joint
Conference on Artificial Intelligence, pages 8151–8159, 2024.
[22] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang
Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language
models with instruction tuning. In Thirty-seventh Conference on Neural Information Processing
Systems, 2023.
[23] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang,
Zhiyong Wang, Lu Sheng, Lei Bai, et al. Lamm: Language-assisted multi-modal instruction-
tuning dataset, framework, and benchmark. Advances in Neural Information Processing Systems,
36, 2024.
[24] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang,
Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large
language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
[25] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing
Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception
of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
11


---

## Page 12

[26] Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek
Mittal, and Peter Henderson. Safety alignment should be made more than just a few tokens
deep. In The Thirteenth International Conference on Learning Representations, 2025.
[27] Vaidehi Patil, Yi-Lin Sung, Peter Hase, Jie Peng, Tianlong Chen, and Mohit Bansal. Unlearn-
ing sensitive information in multimodal LLMs: Benchmark and attack-defense evaluation.
Transactions on Machine Learning Research, 2024.
[28] Tianle Gu, Zeyang Zhou, Kexin Huang, Liang Dandan, Yixu Wang, Haiquan Zhao, Yuanqi Yao,
Yujiu Yang, Yan Teng, Yu Qiao, et al. Mllmguard: A multi-dimensional safety evaluation suite
for multimodal large language models. Advances in Neural Information Processing Systems,
37, 2024.
[29] Peng Qi, Zehong Yan, Wynne Hsu, and Mong Li Lee. Sniffer: Multimodal large language
model for explainable out-of-context misinformation detection. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 13052–13062, 2024.
[30] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy
Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a
laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.
[31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
[32] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao
Chen. mdpo: Conditional preference optimization for multimodal large language models. arXiv
preprint arXiv:2406.11839, 2024.
[33] Renjie Pi, Tianyang Han, Jianshu Zhang, Yueqi Xie, Rui Pan, Qing Lian, Hanze Dong, Jipeng
Zhang, and Tong Zhang. Mllm-protector: Ensuring mllm’s safety without hurting performance.
In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pages 16012–16027, 2024.
[34] Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung,
James T Kwok, and Yu Zhang. Eyes closed, safety on: Protecting multimodal llms via image-
to-text transformation. In European Conference on Computer Vision, pages 388–404. Springer,
2024.
[35] Zhenting Wang, Shuming Hu, Shiyu Zhao, Xiaowen Lin, Felix Juefei-Xu, Zhuowei Li, Ligong
Han, Harihar Subramanyam, Li Chen, Jianfa Chen, et al. Mllm-as-a-judge for image safety
without human labeling. arXiv preprint arXiv:2501.00192, 2024.
[36] Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin,
Senjie Jin, Yu Qiao, Xuanjing Huang, et al. Spa-vl: A comprehensive safety preference
alignment dataset for vision language model. arXiv preprint arXiv:2406.12030, 2024.
[37] Yiting Qu, Xinyue Shen, Yixin Wu, Michael Backes, Savvas Zannettou, and Yang Zhang.
Unsafebench: Benchmarking image safety classifiers on real-world and ai-generated images.
arXiv preprint arXiv:2405.03486, 2024.
[38] Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. Mm-safetybench: A
benchmark for safety evaluation of multimodal large language models. In European Conference
on Computer Vision, pages 386–403. Springer, 2024.
[39] Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. Mm-safetybench: A
benchmark for safety evaluation of multimodal large language models. In European Conference
on Computer Vision, pages 386–403. Springer, 2025.
[40] Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson.
Universal and transferable
adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.
[41] Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath,
Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language
models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint
arXiv:2209.07858, 2022.
12


---

