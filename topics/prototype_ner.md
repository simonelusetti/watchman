# Prototype-Based Learning for NER

**Topic:** Building label prototypes — vectors representing entity types as a whole — and classifying tokens or spans by proximity to those prototypes rather than by a parametric classifier.

**Papers covered:** 18 papers spanning foundational metric learning, the standard NER baseline, the primary benchmark, and the main methodological families in few-shot NER.

---

## 1. The Setting: Few-Shot NER

Named entity recognition is a sequence labelling task: given tokens $\mathbf{x} = (x_1, \ldots, x_T)$, predict a label $y_t \in \mathcal{Y} \cup \{O\}$ for each position, where $\mathcal{Y}$ is a set of entity types and $O$ denotes non-entity. In the standard supervised setting $\mathcal{Y}_\text{train}$ is fixed and large annotated corpora are available.

In the **few-shot NER** setting, the model must generalise to a target type set $\mathcal{Y}_\text{test}$ with $\mathcal{Y}_\text{test} \cap \mathcal{Y}_\text{train} = \emptyset$, given only $K$ labelled sentences per type at inference time. Evaluation is organised into $N$-way $K$-shot **episodes**: each episode samples $N$ entity types, provides $K$ labelled support sentences per type, and asks the model to annotate a set of query sentences. Boundaries between entity spans must also be recovered, not just type labels.

A prototype classifier handles this setting by representing each type $k$ as a single vector $\mathbf{c}_k \in \mathbb{R}^d$ computed from its $K$ support examples and classifying each query token (or span) by

$$\hat{y} = \arg\min_{k \in \{1,\ldots,N,O\}} d(\mathbf{z}, \mathbf{c}_k),$$

where $\mathbf{z}$ is the encoded query representation and $d$ is a distance function. The central design questions are: (i) how to encode $\mathbf{z}$, (ii) how to aggregate support examples into $\mathbf{c}_k$, (iii) what distance $d$ to use, and (iv) how to handle the $O$-class.

---

## 2. Foundational Metric and Meta-Learning

### 2.1 Prototypical Networks — Snell et al., NeurIPS 2017 · `prototypical_networks_fewshot_learning`

**Core idea.** Learn an embedding function $f_\phi : \mathcal{X} \to \mathbb{R}^d$ such that classes form compact, well-separated clusters. Each class $k$ is represented by the centroid of its embedded support points:

$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(\mathbf{x}_i, y_i) \in S_k} f_\phi(\mathbf{x}_i).$$

**Classification.** Query $\hat{\mathbf{x}}$ is classified via softmax over negative squared Euclidean distances:

$$p_\phi(y = k \mid \hat{\mathbf{x}}) = \frac{\exp\!\bigl(-\|\,f_\phi(\hat{\mathbf{x}}) - \mathbf{c}_k\|^2\bigr)}{\sum_{k'} \exp\!\bigl(-\|\,f_\phi(\hat{\mathbf{x}}) - \mathbf{c}_{k'}\|^2\bigr)}.$$

**Training.** Episodes are sampled from the training classes. Within each episode $\mathcal{E}$, a subset of classes forms the support set and a disjoint subset of examples forms the query set. The loss is:

$$\mathcal{L} = -\frac{1}{|\mathcal{Q}|}\sum_{(\hat{\mathbf{x}}, y) \in \mathcal{Q}} \log p_\phi(y \mid \hat{\mathbf{x}}),$$

where $\mathcal{Q}$ is the query set of the episode. The encoder $f_\phi$ is the only set of learned parameters; the prototype $\mathbf{c}_k$ is computed at runtime with no gradient storage across episodes.

**Theoretical grounding.** The authors prove that using the mean with squared Euclidean distance is the *unique* choice consistent with Bregman divergence geometry: for any regular Bregman divergence $d_\psi(\mathbf{z}, \boldsymbol{\mu}) = \psi(\mathbf{z}) - \psi(\boldsymbol{\mu}) - \langle \nabla\psi(\boldsymbol{\mu}), \mathbf{z} - \boldsymbol{\mu}\rangle$, the minimiser of the expected divergence $\mathbb{E}[d_\psi(\mathbf{z}, \boldsymbol{\mu})]$ is the mean $\boldsymbol{\mu} = \mathbb{E}[\mathbf{z}]$. For squared Euclidean distance ($\psi(\mathbf{z}) = \|\mathbf{z}\|^2$) this gives the mean directly. Cosine distance is shown empirically to perform worse because it does not satisfy Bregman structure.

**Results.** On miniImageNet, 5-way 1-shot: **49.42 ± 0.78%**, 5-way 5-shot: **68.20 ± 0.66%**, outperforming Matching Networks at the time. On Omniglot, 5-way 1-shot: **98.8%**, 5-way 5-shot: **99.7%**.

---

### 2.2 Matching Networks — Vinyals et al., NeurIPS 2016 · `matching_networks_one_shot_learning`

**Core idea.** Classify by a soft attention-weighted sum over all labelled support instances, rather than collapsing them to a class mean. Both the query and support encoders are made context-aware via an LSTM that conditions on the entire support set.

**Classification.** Given support set $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^k$, the prediction for query $\hat{\mathbf{x}}$ is:

$$\hat{y} = \sum_{i=1}^k a(\hat{\mathbf{x}}, \mathbf{x}_i)\, y_i,$$

where the attention kernel is:

$$a(\hat{\mathbf{x}}, \mathbf{x}_i) = \frac{\exp\!\bigl(c(f(\hat{\mathbf{x}}),\, g(\mathbf{x}_i))\bigr)}{\sum_j \exp\!\bigl(c(f(\hat{\mathbf{x}}),\, g(\mathbf{x}_j))\bigr)},$$

with $c(\cdot, \cdot)$ being cosine similarity. This is a weighted $k$-NN rule with learned distances.

**Full Context Embeddings (FCE).** The key novelty: support encoder $g$ is a bidirectional LSTM that reads the full support set $S$, so that $g(\mathbf{x}_i)$ is conditioned on all other support examples:

$$\overrightarrow{h}_i, \overrightarrow{c}_i = \text{LSTM}(f'(\mathbf{x}_i), \overrightarrow{h}_{i-1}, \overrightarrow{c}_{i-1}), \quad g(\mathbf{x}_i) = [\overrightarrow{h}_i; \overleftarrow{h}_i] + f'(\mathbf{x}_i).$$

The query encoder $f$ uses an attention LSTM over the support embeddings with $K$ read steps:

$$\hat{h}_t, \hat{c}_t = \text{LSTM}(f'(\hat{\mathbf{x}}), [h_{t-1}; r_{t-1}], \hat{c}_{t-1}),$$
$$r_{t-1} = \sum_i a_{t-1,i}\, g(\mathbf{x}_i), \quad a_{t-1,i} \propto \exp(\hat{h}_{t-1}^\top g(\mathbf{x}_i)).$$

**Training.** Episodic, minimising cross-entropy with the same episode structure as Prototypical Networks.

**Results.** miniImageNet, 5-way 1-shot: **43.56 ± 0.84%**, 5-way 5-shot: **55.31 ± 0.73%**. This underperforms Prototypical Networks in the 5-shot case because the per-instance attention is expensive and prone to noise with $k$ small, while the prototype mean is already a good sufficient statistic.

---

### 2.3 MAML — Finn, Abbeel & Levine, ICML 2017 · `modelagnostic_metalearning_fast_adaptation_deep_networks`

**Core idea.** Find a parameter initialisation $\theta$ such that a small number of gradient steps on any new task's loss achieves good performance. Unlike metric methods, MAML does not learn a metric space; it learns where to initialise gradient descent.

**Inner loop.** For task $\mathcal{T}_i$ with support set $\mathcal{D}^\text{tr}_i$, one gradient step produces adapted parameters:

$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta;\, \mathcal{D}^\text{tr}_i).$$

With $m$ steps this generalises to:

$$\theta^{(m)}_i = \theta^{(m-1)}_i - \alpha \nabla_{\theta^{(m-1)}_i} \mathcal{L}_{\mathcal{T}_i}(f_{\theta^{(m-1)}_i}).$$

**Meta-objective.** Minimise the post-adaptation loss across tasks, evaluated on each task's query set $\mathcal{D}^\text{te}_i$:

$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i};\, \mathcal{D}^\text{te}_i).$$

**Meta-gradient.** The outer gradient requires backpropagating through the inner gradient step:

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i}) = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i} \cdot \frac{\partial \theta'_i}{\partial \theta} = \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i} \cdot \Bigl(I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)\Bigr),$$

involving the Hessian $\nabla^2_\theta \mathcal{L}$. The first-order approximation (FOMAML) drops the Hessian term, setting $\frac{\partial \theta'_i}{\partial \theta} \approx I$, and achieves nearly identical performance at much lower cost.

**Results.** miniImageNet, 5-way 1-shot: **48.70 ± 1.84%**, 5-way 5-shot: **63.11 ± 0.92%**. Comparable to Prototypical Networks, but MAML is strictly more flexible: it can incorporate any amount of labelled data at test time through additional inner-loop steps, whereas prototype methods are bounded by the expressiveness of the metric space.

---

### 2.4 Relation Network — Sung et al., CVPR 2018 · `learning_compare_relation_network_fewshot_learning`

**Core idea.** The distance function itself is a learned neural network, not a fixed metric. This allows non-metric, asymmetric, and non-linear comparisons between support and query.

**Architecture.** Two modules: embedding function $f_\phi$ and relation module $g_\psi$. The class representation for type $i$ in an $N$-way $K$-shot episode is:

$$\mathbf{c}_i = \sum_{k=1}^K f_\phi(\mathbf{x}^k_i) \in \mathbb{R}^D.$$

The relation score for query $\hat{\mathbf{x}}$ against class $i$ is:

$$r_i = g_\psi\!\bigl(\mathcal{C}(\mathbf{c}_i,\, f_\phi(\hat{\mathbf{x}}))\bigr) \in [0,1],$$

where $\mathcal{C}$ denotes channel-wise concatenation and $g_\psi$ is a multi-layer network with sigmoid output. Note: element-wise sum (not mean) is used for aggregation; the relation module implicitly normalises by $K$ through its learned weights.

**Loss.** Mean squared error between relation scores and binary targets $\mathbf{1}[i = y]$:

$$\mathcal{L} = \sum_{\hat{\mathbf{x}}, y} \sum_i \bigl(r_i - \mathbf{1}[i = y]\bigr)^2.$$

MSE is used rather than cross-entropy so that relation scores are interpretable as pairwise similarity scores in $[0,1]$.

**Results.** miniImageNet, 5-way 1-shot: **50.44 ± 0.82%**, 5-way 5-shot: **65.32 ± 0.70%** — marginally better than Prototypical Networks in 1-shot. Omniglot 5-way 1-shot: **99.6 ± 0.2%**. The learned distance provides little benefit in 5-shot where simple Euclidean distance already exploits sufficient statistics.

---

### 2.5 Induction Networks — Geng et al., EMNLP 2019 · `induction_networks_fewshot_text_classification`

**Core idea.** The mean prototype is a poor representative when support instances are diverse. Replace it with a learned induction module that routes support instances to a class capsule, down-weighting atypical examples.

**Architecture.** Three components: (i) encoder $f_\phi$ producing instance embeddings $\mathbf{e}_i$; (ii) induction module producing class capsule $\mathbf{c}_k$; (iii) relation module $g_\psi$ scoring (query, class) pairs.

**Induction module — dynamic routing.** The class capsule is computed over $R$ routing iterations:

$$\mathbf{u}_i = \mathbf{W}\mathbf{e}_i, \quad \text{(linear squash projection)}$$
$$\mathbf{c}_k^{(r)} = \text{squash}\Bigl(\sum_i c_{ik}^{(r)} \mathbf{u}_i\Bigr), \quad \text{squash}(\mathbf{v}) = \frac{\|\mathbf{v}\|^2}{1+\|\mathbf{v}\|^2} \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|},$$

where routing coefficients are updated via agreement:

$$b_{ik}^{(r+1)} \leftarrow b_{ik}^{(r)} + \langle \mathbf{u}_i, \mathbf{c}_k^{(r)}\rangle, \quad c_{ik}^{(r)} = \frac{\exp(b_{ik}^{(r)})}{\sum_{i'} \exp(b_{i'k}^{(r)})}.$$

Instances $\mathbf{u}_i$ that point in the same direction as the current capsule receive higher weight; those that deviate are down-weighted. After $R=3$ iterations the capsule $\mathbf{c}_k$ concentrates on the coherent core of the support set.

**Relation scoring.** Cosine similarity after non-linear projection: $r = g_\psi(\text{cos}(\mathbf{q}, \mathbf{c}_k))$ for query $\mathbf{q} = f_\phi(\hat{\mathbf{x}})$.

**Results.** On SNIPS (intent classification), 5-way 1-shot: **76.39%** vs. Prototypical Networks **69.19%** (+7.2 pp); 5-way 5-shot: **79.15%** vs. **77.37%**. On Amazon Reviews, 5-way 1-shot: **58.70%** vs. **53.05%** (+5.7 pp). Gains are largest in 1-shot, consistent with the routing mechanism being most useful when few support examples mean the mean is unreliable.

---

## 3. Few-Shot NER: Methods

### 3.1 Neural NER Baseline — Lample et al., NAACL 2016 · `neural_architectures_named_entity_recognition`

**The fully supervised reference point.** BiLSTM-CRF is the architecture that defines NER performance before pre-trained transformers and against which few-shot methods are implicitly compared.

**Encoding.** For each token $x_t$, the representation is the concatenation of a pre-trained word embedding $\mathbf{w}_t \in \mathbb{R}^{d_w}$ and a character-level representation $\mathbf{c}_t \in \mathbb{R}^{d_c}$ produced by a CNN or LSTM over the character sequence of $x_t$:

$$\mathbf{x}_t = [\mathbf{w}_t;\, \mathbf{c}_t].$$

A bidirectional LSTM then contextualises:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_\rightarrow(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1}), \quad \overleftarrow{\mathbf{h}}_t = \text{LSTM}_\leftarrow(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1}), \quad \mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t;\, \overleftarrow{\mathbf{h}}_t].$$

**CRF output layer.** A linear-chain CRF scores the full label sequence $\mathbf{y} = (y_1,\ldots,y_T)$ by:

$$s(\mathbf{x}, \mathbf{y}) = \sum_{t=1}^T \bigl(\mathbf{W}_{y_t}^\top \mathbf{h}_t + T_{y_{t-1}, y_t}\bigr),$$

where $\mathbf{W} \in \mathbb{R}^{|\mathcal{Y}| \times 2d}$ are emission weights and $T \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|}$ is a learned transition matrix. The conditional probability is:

$$p(\mathbf{y} \mid \mathbf{x}) = \frac{\exp(s(\mathbf{x}, \mathbf{y}))}{\sum_{\mathbf{y}'} \exp(s(\mathbf{x}, \mathbf{y}'))},$$

where the denominator is computed via the forward algorithm in $O(T|\mathcal{Y}|^2)$. Training minimises $-\log p(\mathbf{y}^\star \mid \mathbf{x})$; inference uses Viterbi decoding.

**Why the transition matrix matters.** Without $T$, the model applies a separate softmax at each position and can produce sequences like $O \to I\text{-PER}$ (an inside tag following a non-entity). The CRF penalises such transitions by learning $T_{O, I\text{-PER}} \ll 0$, yielding +1.79 F1 on CoNLL-2003 English vs. the BiLSTM-softmax baseline.

**Results.** CoNLL-2003 English: **90.94 F1**; German: **78.76 F1**; CoNLL-2002 Spanish: **85.75 F1**; Dutch: **81.74 F1** — all without language-specific resources or gazetteers.

---

### 3.2 Few-NERD Benchmark — Ding et al., ACL 2021 · `fewnerd_fewshot_named_entity_recognition_dataset`

**Why a new dataset was needed.** Prior few-shot NER work resampled from CoNLL-2003 (4 entity types) into episodes. With only 4 types, a 5-way episode is impossible, and any 2-way episode trivially covers most of the type space. Generalisation to genuinely novel types cannot be measured.

**Dataset construction.** 188,238 Wikipedia sentences, 4.6M tokens, annotated with an 8-coarse / 66-fine-grained entity type hierarchy. Each token is labelled in BIO format. The hierarchy enables both coarse and fine-grained episode sampling: coarse types include PERSON, LOCATION, ORGANIZATION; fine-grained types include ACTOR, DIRECTOR, CITY, RIVER, COMPANY, and so on. Inter-annotator agreement is high (Fleiss' $\kappa > 0.8$).

**Episode structure.** Each $N$-way $K$-shot episode samples $N$ fine-grained entity types, provides $K$ support sentences per type with BIO labels, and provides a set of unlabelled query sentences. Non-entity tokens in support and query are all labelled $O$. Two evaluation settings:
- **Intra:** support and query sentences are drawn from the same coarse domain.
- **Inter:** support and query sentences may be drawn from different coarse domains, requiring more robust type abstraction.

**Baseline results on Few-NERD (5-way 1-shot).** From the paper's own benchmarks:

| Model | Intra F1 | Inter F1 |
|---|---|---|
| ProtoBERT | 41.4 | 19.8 |
| NNShot | 41.4 | 26.0 |
| StructShot | 43.7 | 25.3 |
| Supervised (upper bound) | 68.8 | 52.9 |

The gap between the best few-shot model and the supervised upper bound is large — particularly on the inter split — establishing Few-NERD as a genuinely open benchmark.

---

### 3.3 NNShot / StructShot — Yang & Katiyar, EMNLP 2020 · `simple_effective_fewshot_named_entity_recognition`

**Motivation.** Meta-learning objectives push the encoder away from the BERT geometry that is already well-suited for NER. A non-parametric approach that simply reuses pretrained BERT representations may be stronger.

**NNShot.** A BERT-CRF model is trained on source-domain NER data as usual. At test time, its parameters are frozen. Each support token $(x_s, y_s)$ is encoded to $f(x_s) \in \mathbb{R}^d$ (the BERT hidden state). Each query token $x_t$ is classified by cosine nearest-neighbour over the support:

$$\hat{y}_t = \arg\min_{k} \min_{s:\, y_s = k} \frac{f(x_t) \cdot f(x_s)}{\|f(x_t)\|\|f(x_s)\|}.$$

This is a 1-nearest-neighbour rule: the query token inherits the label of its single closest support token, regardless of which class that token belongs to.

**StructShot.** NNShot classifies each token independently, ignoring BIO consistency. StructShot adds sequential structure via Viterbi decoding. The emission score at position $t$ for label $k$ is the NNShot distance $e_{tk} = \min_{s: y_s = k} d(f(x_t), f(x_s))$. The transition matrix is the **abstract transition matrix** $\tilde{T}$, which encodes only BIO structural constraints:

$$\tilde{T}_{k, k'} = \begin{cases} 0 & \text{if transition } k \to k' \text{ is valid in BIO} \\ -\infty & \text{otherwise} \end{cases}$$

plus a learned scalar offset per structural transition type (e.g., $O \to B$, $B\text{-X} \to I\text{-X}$, $B\text{-X} \to B\text{-Y}$). This matrix has no entity-type-specific parameters, only structural ones, making it directly transferable to any episode. Viterbi decoding then finds:

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \sum_t e_{t,y_t} + \sum_t \tilde{T}_{y_{t-1}, y_t}.$$

**Results.** On CoNLL-2003 English, 1-shot (transferring from OntoNotes):

| Model | F1 |
|---|---|
| Prototypical-BERT | 70.1 |
| NNShot | 77.3 |
| StructShot | **83.1** |

On OntoNotes, 1-shot (transferring from CoNLL):

| Model | F1 |
|---|---|
| Prototypical-BERT | 51.2 |
| NNShot | 56.6 |
| StructShot | **61.2** |

StructShot's improvement over NNShot is entirely from the Viterbi step — the transition prior alone provides +5–6 F1 at zero additional training cost.

---

### 3.4 Comprehensive Study — Huang et al., EMNLP 2021 · `fewshot_named_entity_recognition_comprehensive_study`

**Experimental design.** Three axes studied across 10 NER datasets (general, biomedical, scientific domains): (1) meta-learning with prototypes, (2) entity-focused noisy pre-training, (3) self-training on unlabelled in-domain text.

**Prototype meta-learning (ProtoBERT).** For a $K$-shot episode, the prototype for type $k$ is:

$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{t:\, y_t = k,\, (\mathbf{x}, \mathbf{y}) \in S} f_\theta(\mathbf{x})_t \in \mathbb{R}^d,$$

the mean over support tokens of type $k$ across all support sentences. A query token at position $t$ is classified by:

$$\hat{y}_t = \arg\min_{k \in \{1,\ldots,N,O\}} \|f_\theta(\mathbf{x})_t - \mathbf{c}_k\|^2,$$

with the $O$-prototype $\mathbf{c}_O$ built analogously from all non-entity support tokens. Training minimises the episodic prototypical cross-entropy.

**Noisy pre-training.** A standard token classifier is pre-trained on weakly-labelled entity data (Wikipedia anchors linked to Wikidata entity types), then fine-tuned in the few-shot episodes. This provides entity-rich representations before episodic training.

**Self-training.** The few-shot model generates pseudo-labels for unlabelled in-domain sentences, selects high-confidence predictions, adds them to the training set, and iterates. The confidence threshold is the mean model probability over the episode query set.

**Key quantitative findings** (averaged across 10 datasets, $K=5$):

| Strategy | Avg. F1 |
|---|---|
| Baseline (BERT fine-tune, no meta-learning) | 42.1 |
| Meta-learning (ProtoBERT) | 47.8 |
| Noisy pre-training | 49.5 |
| Self-training | 51.3 |
| All three combined | **56.4** |

At $K=1$, meta-learning has the largest individual contribution (+7.2 F1 over baseline). By $K=20$, direct fine-tuning matches prototypical meta-learning, establishing the regime where prototypes are most valuable.

---

### 3.5 CONTaiNER — Das et al., ACL 2022 · `container_fewshot_named_entity_recognition_contrastive`

**Motivation.** Prototype methods are category-specific: the representation of each entity type is built from support tokens of that type. If the training objective is parameterised by class indices, the model may not generalise to unseen type vocabularies. CONTaiNER trains a category-agnostic objective.

**Gaussian token embeddings.** Each token $x_t$ is encoded by BERT to $\mathbf{h}_t \in \mathbb{R}^d$, then projected to mean $\boldsymbol{\mu}_t \in \mathbb{R}^d$ and diagonal log-variance $\log\boldsymbol{\sigma}^2_t \in \mathbb{R}^d$ via linear layers. The token distribution is $q_t = \mathcal{N}(\boldsymbol{\mu}_t, \text{diag}(\boldsymbol{\sigma}^2_t))$.

**Similarity via KL divergence.** For diagonal Gaussians, the KL divergence has a closed form:

$$D_\text{KL}(q_i \| q_j) = \frac{1}{2}\sum_{l=1}^d \Bigl(\frac{\sigma^2_{i,l}}{\sigma^2_{j,l}} + \frac{(\mu_{i,l} - \mu_{j,l})^2}{\sigma^2_{j,l}} - 1 + \log\frac{\sigma^2_{j,l}}{\sigma^2_{i,l}}\Bigr).$$

The symmetrised similarity is $\text{sim}(i,j) = -\frac{1}{2}(D_\text{KL}(q_i\|q_j) + D_\text{KL}(q_j\|q_i))$.

**Contrastive loss.** Within each episode, token pairs $(i, j)$ with $y_i = y_j$ are positives; pairs with $y_i \neq y_j$ are negatives. The NT-Xent (InfoNCE) loss over token $i$:

$$\mathcal{L}_i = -\log \frac{\sum_{j: y_j = y_i} \exp(\text{sim}(i,j)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(i,j)/\tau)}.$$

Crucially, no class index appears in the loss — only the binary same/different-label relation. This makes the objective transferable to any episode's type vocabulary at inference time.

**Inference.** Query token $x_t$ is classified by nearest-neighbour in KL space over the support set: $\hat{y}_t = \arg\min_k \min_{s: y_s = k} D_\text{KL}(q_t \| q_s)$.

**Results** on Few-NERD (micro-F1):

| Setting | NNShot | ProtoBERT | CONTaiNER |
|---|---|---|---|
| 5-way 1-shot intra | 41.4 | 41.4 | **59.9** |
| 5-way 1-shot inter | 26.0 | 19.8 | **31.1** |
| 5-way 5-shot intra | 53.3 | 53.3 | **66.4** |
| 5-way 5-shot inter | 36.3 | 36.7 | **43.7** |

CONTaiNER improves over the previous best by 3–13 F1 points, with the largest absolute gain on intra (where the episode distribution is simpler and the Gaussian uncertainty proves most useful).

---

### 3.6 Decomposed Meta-Learning — Ma et al., ACL Findings 2022 · `decomposed_metalearning_fewshot_named_entity_recognition`

**Motivation.** Span detection (which positions are entities?) is largely type-agnostic and benefits from source-domain supervision. Entity typing (which type?) is type-specific and benefits from a prototype metric space. Solving both jointly in a single episodic objective conflates two different learning signals.

**Few-shot span detection with MAML.** A BERT-based binary sequence labeller $f^\text{span}_\theta$ assigns each token a probability of being part of an entity. MAML is applied: the meta-initialisation $\theta^\text{span}$ is found by minimising the cross-entropy span detection loss after one inner gradient step:

$$\theta^\text{span}{'}_i = \theta^\text{span} - \alpha\nabla_{\theta^\text{span}}\mathcal{L}^\text{span}_i, \quad \min_{\theta^\text{span}} \sum_i \mathcal{L}^\text{span}_i(f^\text{span}_{\theta^\text{span}{'}_{i}}).$$

The binary supervision (entity vs. non-entity) is type-agnostic, so the meta-initialisation can draw on all source-domain entity annotations regardless of type.

**Few-shot entity typing with MAML-ProtoNet.** Detected spans are encoded by a separate BERT encoder $f^\text{type}_\theta$. Prototypes are built as span-representation means:

$$\mathbf{c}_k = \frac{1}{|S_k|}\sum_{(i,j)\in S_k} f^\text{type}_\theta(\mathbf{x}, i, j) \in \mathbb{R}^d,$$

where $f^\text{type}_\theta(\mathbf{x}, i, j)$ is the pooled representation of span $(i,j)$ in sentence $\mathbf{x}$. The encoder initialisation is found by MAML applied to the episodic prototypical cross-entropy loss. This is MAML-ProtoNet: MAML finds an encoder from which the prototype metric space is immediately useful with no fine-tuning.

**Results** on Few-NERD inter (micro-F1):

| Model | 5-way 1-shot | 5-way 5-shot |
|---|---|---|
| ProtoBERT | 19.8 | 36.7 |
| CONTaiNER | 31.1 | 43.7 |
| Decomposed | **35.8** | **46.4** |

Ablations show that MAML on span detection contributes +2.3 F1 independently, and MAML-ProtoNet adds a further +2.4 F1 at $K=1$.

---

### 3.7 SpanProto — Wang et al., EMNLP 2022 · `spanproto_twostage_spanbased_prototypical_network_fewshot`

**Motivation.** Token-level prototypes aggregate representations of tokens that lie within entity spans alongside boundary tokens, diluting the entity signal. Prototypes built at the span level represent semantically coherent units.

**Stage 1 — span selection.** Every candidate span $(i, j)$ with $j - i < L$ is scored by a boundary classifier. Given BERT contextualised representations $\mathbf{h}_1, \ldots, \mathbf{h}_T$:

$$s(i,j) = \sigma\bigl(\mathbf{w}^\top_s \mathbf{h}_i + \mathbf{w}^\top_e \mathbf{h}_j + \mathbf{w}^\top_\Delta (\mathbf{h}_j - \mathbf{h}_i)\bigr),$$

where $\mathbf{w}_s, \mathbf{w}_e, \mathbf{w}_\Delta \in \mathbb{R}^d$ are learned. Spans with $s(i,j) > \rho$ (threshold) pass to Stage 2. The stage-1 loss is binary cross-entropy over all candidate spans.

**Stage 2 — span typing.** The span representation is the mean of its token representations plus a width embedding:

$$\mathbf{v}_{ij} = \text{MLP}\Bigl(\Bigl[\frac{1}{j-i+1}\sum_{t=i}^j \mathbf{h}_t;\, \mathbf{e}_{j-i}\Bigr]\Bigr),$$

where $\mathbf{e}_{j-i}$ is a learned span-width embedding. Entity type $k$ prototype:

$$\mathbf{c}_k = \frac{1}{|S_k|}\sum_{(i,j)\in S_k}\mathbf{v}_{ij}.$$

A non-entity prototype is built from negative support spans: $\mathbf{c}_O = \frac{1}{|S_O|}\sum_{(i,j)\in S_O}\mathbf{v}_{ij}$ where $S_O$ contains spans that are annotated as non-entities (sampled from background tokens).

**Classification** by nearest prototype with Euclidean distance: $\hat{k} = \arg\min_{k \in \{1,\ldots,N,O\}} \|\mathbf{v}_{ij} - \mathbf{c}_k\|^2$.

**Joint training.** The full objective is $\mathcal{L} = \mathcal{L}^\text{detect} + \lambda\mathcal{L}^\text{type}$ where $\mathcal{L}^\text{type}$ is the prototypical cross-entropy over span–type assignments.

**Results** on Few-NERD (micro-F1):

| Model | 5-way 1-shot intra | 5-way 1-shot inter | 5-way 5-shot intra | 5-way 5-shot inter |
|---|---|---|---|---|
| CONTaiNER | 59.9 | 31.1 | 66.4 | 43.7 |
| Decomposed | — | 35.8 | — | 46.4 |
| SpanProto | **63.7** | **38.1** | **70.2** | **50.0** |

---

### 3.8 EP-Net — Su et al., COLING 2022 · `fewshot_named_entity_recognition_entitylevel_prototypical`

**Motivation.** In high-dimensional spaces with small support sets, prototypes of distinct entity types can occupy nearby regions (prototype collapse). Standard episodic training does not explicitly penalise this.

**Span encoder.** Like SpanProto, EP-Net builds span representations $\mathbf{v}_{ij}$ and prototypes $\mathbf{c}_k = \frac{1}{|S_k|}\sum \mathbf{v}_{ij}$. The encoder is fine-tuned exclusively on span-level supervision (no token-level labels).

**Dispersion loss.** For $N$ entity types in an episode with prototypes $\{\mathbf{c}_k\}$, the dispersion objective maximises pairwise distances:

$$\mathcal{L}_\text{disp} = -\sum_{k \neq k'} \log \sigma\!\bigl(d(\mathbf{c}_k, \mathbf{c}_{k'}) - m\bigr),$$

where $d$ is Euclidean distance, $m$ is a margin hyperparameter, and $\sigma$ is the sigmoid. This is a soft hinge: when $d(\mathbf{c}_k, \mathbf{c}_{k'}) > m$, the term is near 0; when prototypes collapse below margin $m$, the loss pushes them apart.

**Full objective:**

$$\mathcal{L} = \mathcal{L}_\text{proto} + \lambda\,\mathcal{L}_\text{disp},$$

where $\mathcal{L}_\text{proto}$ is the standard prototypical cross-entropy over span classifications.

**Results** on Few-NERD (micro-F1, improvements over ProtoBERT):

| Setting | ProtoBERT | EP-Net | $\Delta$ |
|---|---|---|---|
| 5-way 1-shot intra | 41.4 | 52.1 | +10.7 |
| 5-way 1-shot inter | 19.8 | 28.4 | +8.6 |
| 5-way 5-shot intra | 53.3 | 62.8 | +9.5 |
| 5-way 5-shot inter | 36.7 | 43.2 | +6.5 |

The dispersion loss contributes approximately +3–4 F1 on top of the span-level representation alone.

---

### 3.9 MeTNet — Dong et al., 2023 · `metalearning_triplet_network_adaptive_margins_fewshot`

**Motivation.** A fixed global margin in a contrastive or dispersion loss cannot account for the variable semantic distances between entity types in different episodes. Types like CITY and COUNTRY should be pushed farther apart than CITY and ACTOR.

**Adaptive margin computation.** Each entity type name (e.g., "city", "country") is encoded by a lightweight encoder $h_\phi$ to a type-name vector $\mathbf{t}_k = h_\phi(\text{name}_k)$. The margin between types $k$ and $k'$ is:

$$m_{k,k'} = m_0 \cdot \Bigl(1 + \gamma\,\text{cos}(\mathbf{t}_k, \mathbf{t}_{k'})\Bigr),$$

where $m_0$ is a base margin, $\gamma > 0$ scales the cosine contribution. Semantically similar types (high cosine) receive larger margins; dissimilar types receive smaller ones.

**Triplet loss with adaptive margins.** Entity type prototypes $\mathbf{c}_k$ are built as support means. For anchor $a$ of type $k$, positive $p$ also of type $k$, and negative $n$ of type $k'$:

$$\mathcal{L}_\text{triplet} = \sum_{(a,p,n)} \max\!\bigl(0,\; d(f(a), f(p)) - d(f(a), f(n)) + m_{k,k'}\bigr),$$

with $d$ being Euclidean distance. $O$-class tokens are excluded from this loss.

**O-class handling.** A binary classifier $b_\psi(f(x_t)) \in [0,1]$ is trained separately to detect entity tokens. At inference, a token is classified as $O$ if $b_\psi < 0.5$; otherwise it is assigned the type of its nearest prototype. This prevents $O$-tokens from distorting prototype geometry.

**Results** on Few-NERD (micro-F1, compared against CONTaiNER):

| Setting | CONTaiNER | MeTNet | $\Delta$ |
|---|---|---|---|
| 5-way 1-shot inter | 31.1 | 35.2 | +4.1 |
| 5-way 5-shot inter | 43.7 | 47.8 | +4.1 |
| 10-way 1-shot inter | 21.4 | 26.9 | +5.5 |

Gains are largest in the 10-way setting, where the adaptive margins matter most because the episode contains more potentially confusable type pairs.

---

## 4. Description-Driven Prototype Methods

In the methods above, the prototype $\mathbf{c}_k$ is built from labelled support examples. An alternative is to represent entity types as encodings of their **natural language descriptions**, enabling zero-shot inference.

### 4.1 Leveraging Type Descriptions — Aly et al., ACL 2021 · `leveraging_type_descriptions_zeroshot_named_entity`

**MRC formulation.** Entity type description $d_k$ (e.g., "a geopolitical entity such as a country, city, or state") is the question; input sentence $\mathbf{x}$ is the passage. A cross-encoder BERT takes the concatenation $[\text{[CLS]}\, d_k\, \text{[SEP]}\, \mathbf{x}\, \text{[SEP]}]$ and produces start and end logits over token positions via two linear heads:

$$p^\text{start}_t = \text{softmax}(\mathbf{w}^\text{start} \cdot \mathbf{h}_t), \quad p^\text{end}_t = \text{softmax}(\mathbf{w}^\text{end} \cdot \mathbf{h}_t).$$

The predicted entity span for type $k$ is $(\hat{i}, \hat{j}) = \arg\max_{i \leq j} p^\text{start}_i \cdot p^\text{end}_j$, subject to the constraint $j - i < L_\text{max}$.

**Entailment formulation.** Alternatively, for each candidate span $s$, a yes/no classifier scores the entailment $P(\text{type}(s) = k \mid d_k, \mathbf{x}, s)$ via the [CLS] representation of the cross-encoded input.

**O-class threshold.** Since the O-class has no fixed description, the model applies type $k$ only if $\max_i p^\text{start}_i \cdot p^\text{end}_i > \tau_k$, where $\tau_k$ is a per-type threshold estimated on a development set. Spans below all thresholds are labelled $O$.

**Results** on OntoNotes (zero-shot, tested on the 10 rarest entity types):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Zero-shot text clf. baseline | 39.1 | 28.4 | 32.9 |
| MRC-based (this paper) | 51.2 | 44.7 | **47.7** |
| Entailment-based | 48.3 | 41.6 | 44.7 |

Description quality matters significantly: replacing human-written descriptions with single-word label names drops F1 by ~8 points.

---

### 4.2 SpanNER — Wang et al., EMNLP Findings 2021 · `learning_language_description_lowshot_named_entity`

**Architecture.** A single BERT encoder is shared for both span representations and type description representations. This forces the two embedding spaces to be aligned without a separate cross-modal projection.

**Span detection.** All spans $(i,j)$ with $j - i < L$ are enumerated. The boundary score uses a bilinear interaction between start and end positions:

$$s_\text{det}(i,j) = \mathbf{h}_i^\top \mathbf{W}_b\, \mathbf{h}_j,$$

where $\mathbf{W}_b \in \mathbb{R}^{d \times d}$ is learned. The detection loss is binary cross-entropy over all candidate spans.

**Span typing via contrastive description matching.** The span representation is the mean of its token representations: $\mathbf{v}_{ij} = \frac{1}{j-i+1}\sum_{t=i}^j \mathbf{h}_t$. Each entity type $k$ is represented by encoding its description string $d_k$ through the same BERT encoder and taking the [CLS] token: $\mathbf{t}_k = \text{BERT}(d_k)_\text{[CLS]}$.

The span-type matching loss is InfoNCE:

$$\mathcal{L}_\text{type} = -\log \frac{\exp(\text{cos}(\mathbf{v}_{ij}, \mathbf{t}_{y_{ij}})/\tau)}{\sum_{k=1}^{|\mathcal{Y}|} \exp(\text{cos}(\mathbf{v}_{ij}, \mathbf{t}_k)/\tau)}.$$

The denominator sums over all entity types in the training corpus, not just the episode types, making the objective aware of the full type space during training.

**Zero-shot inference.** For a new type $k_\text{new}$, encode its description to get $\mathbf{t}_{k_\text{new}}$. No support examples or fine-tuning needed.

**Results** (improvements over best baseline):

| Setting | Improvement |
|---|---|
| Few-shot (5-way 1-shot, OntoNotes) | +10% avg. F1 |
| Domain transfer (CoNLL → WNUT) | +23% avg. F1 |
| Zero-shot (unseen types) | +26% avg. F1 |

---

### 4.3 Label Semantics — Ma et al., ACL Findings 2022 · `label_semantics_few_shot_named_entity`

**Architecture.** Two separate BERT encoders: $f_\text{tok}$ processes the input sentence token by token; $f_\text{lbl}$ processes entity type description strings. No class-specific weight vectors are used.

**Token classification via description matching.** Each token $x_t$ is represented by $f_\text{tok}(\mathbf{x})_t \in \mathbb{R}^d$. Each entity type description $d_k$ (e.g., "a named person, character, or group") is encoded by $f_\text{lbl}$ and mean-pooled to $\mathbf{l}_k \in \mathbb{R}^d$. The probability of type $k$ for token $t$:

$$p(y_t = k) = \frac{\exp(\text{cos}(f_\text{tok}(\mathbf{x})_t,\, \mathbf{l}_k) / \tau)}{\sum_{k'} \exp(\text{cos}(f_\text{tok}(\mathbf{x})_t,\, \mathbf{l}_{k'}) / \tau)}.$$

**Training.** Both encoders are jointly fine-tuned with cross-entropy on labelled NER data. The label encoder receives all type descriptions for the training corpus as its input at each step, so it learns to produce description embeddings that are geometrically aligned with token representations of the corresponding entity type. The dual-encoder allows the token and description spaces to have different inductive biases — token context representations differ statistically from definitional text.

**Why this differs from SpanNER.** SpanNER shares encoder weights, which forces alignment but may constrain the encoding of descriptions (which are definitional) to match the distribution of token contexts (which are in running prose). The dual encoder allows each space to adapt independently.

**Results** (few-shot F1, averaged over OntoNotes + Few-NERD + i2b2):

| $K$ | ProtoBERT | NNShot | Label Semantics |
|---|---|---|---|
| 1-shot | 38.2 | 41.7 | **46.3** |
| 5-shot | 51.4 | 54.8 | **58.1** |
| 10-shot | 58.3 | 60.2 | **63.9** |

Zero-shot (no support examples): **40.2 F1** on OntoNotes rare types, vs. 32.9 for the zero-shot classification baseline.

---

### 4.4 Prompt-Based Metric Learning — Chen et al., ACL Findings 2023 · `promptbased_metric_learning_fewshot_ner`

**Motivation.** Token-level metric methods build prototypes from raw contextualised representations of support tokens. These representations encode the token's sentence context but not the entity type name. Prompting re-encodes support tokens with their label name explicitly in context, injecting label semantics into the prototype.

**Prompt schemas.** Three templates $\{s^{(1)}, s^{(2)}, s^{(3)}\}$ convert a support token $x_t$ with type name $k$ into a new sentence, for example:
- $s^{(1)}$: `"[x_t] is a [type_k]."` — direct assertion
- $s^{(2)}$: `"[type_k]: [context of x_t]"` — type-prefixed context
- $s^{(3)}$: `"The [type_k] is [x_t]."` — inverted assertion

Each template produces a prompt-conditioned BERT encoding $\tilde{\mathbf{h}}_t^{(s)}$. The prompt-aware prototype for schema $s$ and type $k$:

$$\mathbf{c}_k^{(s)} = \frac{1}{|S_k|}\sum_{t \in S_k} \tilde{\mathbf{h}}_t^{(s)}.$$

**Fusion module.** At query time, the original representation $\mathbf{h}_t$ and all prompt-schema representations $\tilde{\mathbf{h}}_t^{(s)}$ are combined via learned attention:

$$\alpha_s = \frac{\exp(\mathbf{w}_s^\top \mathbf{h}_t)}{\sum_{s'} \exp(\mathbf{w}_{s'}^\top \mathbf{h}_t)}, \quad \mathbf{h}_t^\text{fused} = \sum_s \alpha_s\, \tilde{\mathbf{h}}_t^{(s)}.$$

Classification: $\hat{y}_t = \arg\min_k \|\mathbf{h}_t^\text{fused} - \mathbf{c}_k^{(s^*)}\|^2$ where schema $s^*$ is selected by the same attention weights. The training loss is the standard prototypical cross-entropy computed on fused representations.

**Results** on Few-NERD, OntoNotes, and i2b2 (18 settings total; values are micro-F1):

| Setting | Prior SOTA | Prompt-NER | $\Delta$ |
|---|---|---|---|
| Few-NERD 5-way 1-shot intra | 63.7 (SpanProto) | **66.2** | +2.5 |
| Few-NERD 5-way 1-shot inter | 38.1 (SpanProto) | **42.3** | +4.2 |
| OntoNotes 5-way 1-shot | 48.3 | **53.7** | +5.4 |
| i2b2 5-way 1-shot | 41.2 | **49.6** | +8.4 |

State of the art on 16/18 settings, with an average relative F1 improvement of 9.12% and a maximum of 34.51%. Largest gains are at $K=1$ and on fine-grained or domain-shifted settings.

---

## 5. Summary Table

| Paper | Year | Prototype granularity | Type representation | Key mechanism | Primary metric |
|---|---|---|---|---|---|
| Prototypical Networks | 2017 | instance | support mean | Euclidean nearest-mean | accuracy (Omniglot, miniImageNet) |
| Matching Networks | 2016 | instance | per-instance attention | soft attention + FCE LSTM | accuracy (miniImageNet) |
| MAML | 2017 | — | — | gradient initialisation + Hessian | accuracy (miniImageNet) |
| Relation Network | 2018 | instance | support sum | learned distance + MSE loss | accuracy (miniImageNet) |
| Induction Networks | 2019 | class (text) | routed capsule | dynamic routing aggregation | accuracy (SNIPS, Amazon) |
| Lample et al. | 2016 | — | weight matrix | BiLSTM-CRF + Viterbi | F1 (CoNLL-2003) |
| Few-NERD | 2021 | — | — | benchmark dataset | micro-F1 (intra/inter) |
| Comprehensive Study | 2021 | token | support mean | compares meta / pretrain / self-train | F1 (10 datasets) |
| NNShot / StructShot | 2020 | token | per-instance | frozen BERT NN + abstract-$T$ Viterbi | F1 (CoNLL, OntoNotes, WNUT) |
| CONTaiNER | 2022 | token | Gaussian dist. | KL-divergence NT-Xent loss | micro-F1 (Few-NERD) |
| Decomposed | 2022 | span | support mean | MAML-span + MAML-ProtoNet | micro-F1 (Few-NERD) |
| SpanProto | 2022 | span | support mean | two-stage span selection + proto | micro-F1 (Few-NERD) |
| EP-Net | 2022 | span | support mean | dispersion loss (soft hinge) | micro-F1 (Few-NERD) |
| MeTNet | 2023 | span | support mean | triplet + adaptive name-based margin | micro-F1 (Few-NERD) |
| Aly et al. | 2021 | — | type description (cross-enc.) | MRC span extraction / entailment | F1 (OntoNotes, MedMentions) |
| SpanNER | 2021 | span | description (shared enc.) | InfoNCE span–description matching | F1 (OntoNotes, CoNLL, WNUT) |
| Label Semantics | 2022 | token | description (dual enc.) | dual BERT cosine classification | F1 (OntoNotes, Few-NERD, i2b2) |
| Prompt-NER | 2023 | token | prompted support mean | prompt templates + attention fusion | micro-F1 (Few-NERD, OntoNotes, i2b2) |

---

## 6. Design Axes for a Prototype NER System

**Granularity of the prototype unit.** Token-level prototypes (CONTaiNER, ProtoBERT, Prompt-NER) are simple but aggregate over tokens at span boundaries and in varied syntactic roles. Span-level prototypes (SpanProto, EP-Net, Decomposed) represent semantically coherent units at the cost of $O(T^2L)$ candidate enumeration and a separate span detector whose errors propagate.

**Source of type representation.** Support-mean prototypes require labelled examples at inference time. Description-based prototypes (Aly et al., SpanNER, Label Semantics, Prompt-NER) enable zero-shot transfer but depend on description quality and the encoder's ability to bind natural language definitions to entity surface forms in a shared metric space.

**Aggregation function.** The mean minimises expected squared Euclidean distance (Prototypical Networks, Bregman argument) but is sensitive to outlier support instances and to prototype collapse in small episodes. Routing (Induction Networks) is more robust but adds parameters. Dispersion regularisation (EP-Net: soft hinge $-\log\sigma(d - m)$) directly penalises collapse. Adaptive margins (MeTNet) modulate the required separation by type-name similarity.

**Distance function.** Squared Euclidean (Prototypical Networks, SpanProto, EP-Net) is theoretically grounded for mean prototypes. Cosine (Matching Networks, SpanNER, Label Semantics, Prompt-NER) is scale-invariant and works well when norms carry no information. KL divergence over Gaussian distributions (CONTaiNER) provides an asymmetric, uncertainty-aware similarity; the diagonal Gaussian assumption makes it tractable but may underestimate correlations.

**Handling the O-class.** Non-entity tokens are semantically heterogeneous and far outnumber entity tokens. Approaches: (i) build an explicit O-prototype from negative support tokens and classify by nearest of all $N+1$ prototypes (ProtoBERT, SpanProto); (ii) use a distance threshold with no O-prototype (Aly et al.); (iii) train a separate binary entity detector before the metric classifier (MeTNet, Decomposed); (iv) exclude O entirely from the metric loss and handle it post-hoc (CONTaiNER + thresholding).

**Sequential consistency.** All prototype methods classify each token/span independently. Adding BIO transition structure via Viterbi decoding with an abstract transition prior (StructShot: $\tilde{T}$ with no type-specific parameters) consistently adds +5–6 F1 at zero training cost, making it a near-free improvement applicable to any prototype NER system.
