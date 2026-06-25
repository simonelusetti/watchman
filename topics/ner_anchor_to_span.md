# NER: Anchor-to-Span and Two-Stage Proposer–Verifier Paradigms

**Topic:** Moving NER away from per-token sequence labeling toward architectures that first identify candidate spans (anchors, proposals, or scored pairs) and then classify or refine them. Covers span scoring via biaffine models, two-stage object-detection analogies, and boundary offset supervision as an alternative to binary entity/non-entity labels.

**Papers covered:** 4 papers — NER-as-Dependency-Parsing (Yu et al. 2020), NER-as-Object-Detection (Li 2021), BOPN (Tang et al. 2023), End-to-End Entity Detection with Proposer and Regressor (Wen et al. 2023).

---

## 1. Shared Structure: From Tokens to Spans

Sequence labeling encodes each token $x_t$ with a contextual representation $h_t$ and predicts a label $y_t \in \{\text{B-X, I-X, O}\}$ independently (or jointly via CRF). This forces two constraints: each token carries one label (incompatible with nested entities), and entity boundaries are only implicit in the tag scheme. Span-based approaches instead enumerate candidate spans $(i, j)$ as the primary unit and score or classify them directly.

Formally, given a sentence $\mathbf{X} = (x_1, \ldots, x_N)$, span-based NER produces a set of non-overlapping (or hierarchically nested) tuples $\{(s, e, y)\}$ where $s, e$ are start/end token indices and $y \in \mathcal{Y}$ is an entity type. The candidate space is the upper triangular region of the $N \times N$ index matrix, with $N(N+1)/2$ entries. The three papers below differ in *how* this space is scored and how the supervision signal is constructed.

---

## 2. Span Scoring via Biaffine Interaction

### NER as Dependency Parsing — Yu et al., ACL 2020 · `named_entity_recognition_dependency_parsing`

**Core idea.** Adopt the biaffine scorer from graph-based dependency parsing to score all $(i, j)$ index pairs simultaneously, treating NER as a structured prediction problem over span pairs.

**Encoding.** Tokens are encoded by a multi-layer BiLSTM over BERT$_\text{Large}$, fastText, and character CNN embeddings. Two separate FFNNs project each token position into start and end representations:
$$h_s(i) = \mathrm{FFNN}_s(x_{s_i}), \qquad h_e(i) = \mathrm{FFNN}_e(x_{e_i}).$$
Separating start and end projections allows the model to learn that left-boundary context and right-boundary context carry structurally different information.

**Span score.** For each NER category $m$ (including non-entity), the score of candidate span $i$ with start $s_i$ and end $e_i$ is:
$$r_m(i) = h_s(i)^\top \mathbf{U}_m\, h_e(i) + W_m\bigl(h_s(i) \oplus h_e(i)\bigr) + b_m,$$
where $\mathbf{U}_m \in \mathbb{R}^{d \times c \times d}$ is a biaffine weight tensor and $c$ is the number of categories. The full tensor $r_m$ scores all $N^2 \times c$ combinations simultaneously.

**Training and decoding.** Training minimizes softmax cross-entropy over all valid spans. At inference, spans are ranked by their predicted category score and greedily selected subject to a no-overlap (flat NER) or no-clash (nested NER) constraint. A clash between span $i$ and span $j$ is defined as $s_i < s_j \leq e_i < e_j$: partial crossing is disallowed, but containment is permitted for nested NER.

**Results.** State-of-the-art on eight corpora at the time of publication: CoNLL 2003 F1 and ACE 2004/2005 nested NER, with absolute gains of up to 2.2 points over prior best models.

---

## 3. Two-Stage Object Detection Analogy

### NER in the Style of Object Detection — Li 2021 · `named_entity_recognition_style_object_detection`

**Core idea.** Decompose NER into a high-recall region proposal stage and a high-precision verification stage, directly mirroring Faster R-CNN's Region Proposal Network + classification head architecture.

**Stage 1: Entity Region Proposal.** A BERT encoder with a linear head predicts per-token binary probabilities $P(\mathrm{start}_i)$ and $P(\mathrm{end}_i)$ independently. Only the first subword token of each word is scored; trailing subword tokens are masked. Loss is binary cross-entropy applied separately to start and end:
$$\mathcal{L}_s = -\frac{1}{N}\sum_{i=1}^N \Bigl[\mathbb{1}_i^\mathrm{ent}\log p_i^s + (1 - \mathbb{1}_i^\mathrm{ent})\log(1-p_i^s)\Bigr], \quad s \in \{\mathrm{start}, \mathrm{end}\}.$$
The independence between start and end predictions means every $(i, j)$ pair consistent with the length constraint $j - i \leq L_{\max}$ ($L_{\max} \in \{6, 12\}$) is passed to stage 2. The goal is high recall (~98%) at the cost of low precision (~70%).

**Stage 2: Entity Discrimination and Classification.** Each proposal tuple $(\mathrm{sentence},\, i_{\mathrm{start}},\, i_{\mathrm{end}})$ is re-encoded through the same BERT. Max pooling over the region tokens produces a global span representation:
$$z = \max\text{-pool}(\{h_t\}_{t=i_{\mathrm{start}}}^{i_{\mathrm{end}}}).$$
This global view contrasts with sequence labeling, which never aggregates across the full span. Two prediction heads operate on $z$: an entityness head (binary, is this a valid entity?) and a type classification head ($|\mathcal{Y}|$-way, active only for positive proposals). Type loss is conditioned on entityness:
$$\mathcal{L}_{\mathrm{type}} = -\frac{1}{N}\sum_{i=1}^N \mathbb{1}_i^\mathrm{ent}\Bigl[\sum_{c \in \mathcal{Y}} \mathbb{1}_i^c \log p_i^c\Bigr].$$
Two additional boundary losses re-examine the tokens immediately across each boundary using multi-head dot products between start-side and end-side representations, giving the model a "double check" on boundary correctness. The total objective is:
$$\mathcal{L} = \alpha(\mathcal{L}_{\mathrm{start}} + \mathcal{L}_{\mathrm{end}}) + \beta\,\mathcal{L}_{\mathrm{entityness}} + \mathcal{L}_{\mathrm{type}},$$
with $\alpha = 0.5$, $\beta = 1.0$. Unlike object detection, no bounding-box regression is performed — proposals are accepted or rejected, not adjusted.

**Hard negative augmentation.** Precision is substantially improved by adding randomly sampled non-entity spans as hard negatives during stage-2 training, countering the model's tendency to over-trust stage-1 proposals.

**Results.** CoNLL 2003: F1 = 91.8 (BERT-large). OntoNotes 5.0: F1 = 87.0. ACE 2005 nested: F1 = 85.6. GENIA nested: F1 = 76.8. Training/inference speed matches the BERT sequence labeling baseline.

---

## 4. Boundary Offset Supervision

### Boundary Offset Prediction Network — Tang et al., EMNLP Findings 2023 · `boundary_offset_prediction_network_named_entity`

**Core idea.** Replace the binary entity/non-entity label with a signed integer offset $f_s$ indicating how far each candidate span's boundaries are from the nearest true entity of each type. Non-entity spans receive informative gradient instead of a uniform negative signal, directly addressing the severe class imbalance of span enumeration.

**Annotation scheme.** Each candidate span $(i, j)$ is annotated as a quadruple $\{x_i, x_j, f_s, y_m\}$. The offset $f_s \in \{-S, \ldots, -1, 0, 1, \ldots, S\}$ represents the signed displacement from the span's boundary to the nearest entity of type $y_m$: $f_s = 0$ marks a true entity (center span), while $|f_s| = k$ marks a span $k$ positions away. Spans beyond $S$ positions from any entity are labeled "out-of-range" and excluded from entity reconstruction. The label set has cardinality $L = 4S + 2$ when predicting both start and end offsets in parallel.

**Span encoder.** Entity type tokens $\mathbf{P} = \{p_m\}_{m=1}^M$ (one learnable token per category) are prepended to the input sentence and jointly encoded by BERT + one-layer BiLSTM, yielding type representations $\mathbf{H}^Y \in \mathbb{R}^{M \times d}$ and token representations $\mathbf{H}^X \in \mathbb{R}^{N \times d}$.

The span representation for candidate $(i, j)$ is produced by Conditional Layer Normalization (CLN), where the end-token representation modulates the normalization of the start-token:
$$v_{ij} = \mathrm{CLN}(h_i, h_j) = \gamma_j \otimes \mathrm{Norm}(h_i) + \lambda_j,$$
with $\gamma_j = \mathrm{FFN}(h_j)$ and $\lambda_j = \mathrm{FFN}(h_j)$. A learnable region embedding $e_{\mathrm{up}}$ or $e_{\mathrm{low}}$ is appended to distinguish upper- from lower-triangular entries of the candidate matrix: $\hat{v}_{ij} = [v_{ij};\, e_{\mathrm{up}}]$ for $i \leq j$, otherwise $\hat{v}_{ij} = [v_{ij};\, e_{\mathrm{low}}]$.

**Boundary offset predictor.** The biaffine classifier (as in Yu et al. 2020) fuses each entity type representation $h_m$ with each span representation $\hat{v}_{ij}$ after separate FFN projections:
$$h'_y = \mathrm{FFN}(h_y), \qquad v'_{ij} = \mathrm{FFN}(\hat{v}_{ij}),$$
$$c_{mij} = (h'_m)^\top U\, v'_{ij} + W(h'_m \oplus v'_{ij}) + b,$$
yielding a score vector $c_{mij} \in \mathbb{R}^L$ over offset labels for each $(m, i, j)$ triple. The full score tensor $\mathbf{C} \in \mathbb{R}^{M \times N \times N \times L}$ is then processed by 3D convolutions with dilation rates $\{1, 2, 3\}$ to capture quantitative relationships between adjacent span predictions:
$$\mathbf{Q} = \sigma(3\mathrm{DConv}(\mathbf{C})), \qquad \hat{\mathbf{Q}} = \mathrm{Linear}(\mathbf{Q}_1 \oplus \mathbf{Q}_2 \oplus \mathbf{Q}_3).$$
Final offset probabilities: $\hat{o}_{mij} = \mathrm{softmax}(\hat{q}_{mij})$.

**Training objective.** Cross-entropy over all $(m, i, j)$ positions:
$$\mathcal{L} = -\frac{1}{MN^2}\sum_m \sum_i \sum_j o_{mij}^\top \log(\hat{o}_{mij}).$$

**Inference.** Cells $(m, i, j)$ predicted with offset 0 are extracted as entities of type $y_m$. Non-zero in-range offset predictions provide additional evidence that can be aggregated to boost recall. Two heuristic rules prune inconsistent predictions: offsets pointing away from the nearest center span, and offsets violating sequential order with neighbors, are discarded.

**Results.** English flat NER: CoNLL 2003 F1 = 93.19, OntoNotes 5 F1 = 91.16. English nested NER: ACE 2004 F1 = 89.26, ACE 2005 F1 = 90.39, GENIA F1 = 82.14. Chinese flat NER: MSRA F1 = 96.39, Resume NER F1 = 96.78, Weibo NER F1 = 72.92. Ablation confirms that removing boundary offset supervision ($S = 0$) degrades performance on all benchmarks. The model retains 94.5% of its full-data F1 when trained on only 12.5% of available data, versus a sharp decline for the offset-free baseline.

---

## 5. Set Prediction with Iterative Refinement

### End-to-End Entity Detection with Proposer and Regressor — Wen et al. 2023 · `endtoend_entity_detection_proposer_regressor`

**Core idea.** Adopt the DETR set-prediction paradigm from object detection: generate a fixed set of learnable proposals, refine them iteratively, and train with Hungarian matching to eliminate dependence on decoding order. The key departures from prior DETR-for-NER work are (i) a feature-pyramid proposer that initializes proposals from multi-scale contextual representations rather than random query vectors, and (ii) spatially modulated attention in the regressor that focuses each proposal's attention on tokens near its current predicted span.

**Sentence encoder.** Each token $w_i$ is encoded by concatenating character BiGRU pooling $h_i^{\text{char}}$, BERT subword pooling $h_i^{\text{bert}}$, GloVe $h_i^{\text{word}}$, and POS tag $h_i^{\text{pos}}$, then fused by a top-level BiGRU:
$$h_i^t = [h_i^{\text{char}};\, h_i^{\text{bert}};\, h_i^{\text{word}};\, h_i^{\text{pos}}], \qquad H = W\,\text{BiGRU}(H') + b.$$

**Proposer: feature pyramid.** $L$ pyramid layers model spans of increasing size. At layer $l$ with kernel size $k_l$, the receptive field covers $v_l = 1 - l + \sum_{i=1}^l k_i$ tokens. Forward blocks pass information bottom-up via BiGRU + Conv1D; backward blocks pass multi-scale feedback top-down via transposed Conv1D. The feature at each pyramid cell $(i, l)$ covers span $n_{i,l} = [s_{i,l}, e_{i,l}]$.

Proposals are initialized by soft aggregation over all pyramid cells that contain each target token $j$, $\mathcal{N}_j = \{(i,l) \mid j \in [s_{i,l}, e_{i,l}]\}$:
$$r_{i,l} = \text{MLP}(h_{i,l}), \qquad \alpha_{i,l}^j = \frac{\exp(r_{i,l})}{\sum_{(i',l') \in \mathcal{N}_j}\exp(r_{i',l'})}, \qquad n_j = \sum_{(i,l)\in\mathcal{N}_j} \alpha_{i,l}^j \cdot p_{i,l}.$$
The query vector is $q_j = h_{j,1}$ (bottom pyramid layer, most semantically rich). Category logarithms are initialized as $c_j = \text{MLP}(q_j)$.

**Regressor: iterative refinement with spatial attention.** Each of $L$ regressor iterations runs four steps. (1) *Category embedding*: $h'_i = q_i + \alpha_i^\top H^c$ where $\alpha_i = \text{Softmax}(c_i)$ and $H^c$ is a type embedding matrix. (2) *Spatially modulated attention*: a Gaussian-like spatial prior $G_{i,m}(x) = e^{(x-\mu)^\top\Theta(x-\mu)}$ is generated per head from the current span center $\mu_{i,m} = n_i + \Delta n_{i,m}$ and precision matrix $\Theta_{i,m} = \theta_{i,m}\theta_{i,m}^\top$, then added as a log-prior to the attention logits:
$$r_{i,j}^m = \frac{(W_m^q h'_i)^\top(W_m^k h'_j)}{\sqrt{d/M}} + \log G_{i,m}(n_j), \quad \alpha_{i,j}^m = \text{softmax}_j(r_{i,j}^m), \quad o_{i,m} = \sum_j \alpha_{i,j}^m W_m^v h'_j.$$
This biases each proposal to attend to tokens within its current span estimate. (3) *Gated update*: $q_i = \text{Gate}(h'_i,\, \sum_m W_m o_{i,m})$, replacing the FFN with a GRU gate for better expressive capacity. (4) *Logarithm iteration*: $c_i \leftarrow c_i + \text{MLP}(q_i)$ — category distributions are refined at each layer rather than only at the final output.

**Prediction head and span distribution.** The span location probability is produced by a joint pointer network over the refined query and BERT boundary representations:
$$g_i(s,e) = (W_s q_i)^\top(W_s h_s) + (W_e q_i)^\top(W_e h_e), \quad r_i(s,e) = \log G_i(s,e) + g_i(s,e)/\sqrt{d},$$
$$p_i^n = \text{Softmax}(\{r_i(s,e) \mid s \leq e \leq L\}).$$
Type probability: $p_i^c = \text{Softmax}(c_i + \text{MLP}(q_i))$.

**Training: Hungarian matching.** The optimal bijective assignment $\hat\beta$ between $L$ proposals and $N$ entities (padded to $L$ with **None**) is found by the Hungarian algorithm, minimizing total matching cost:
$$\hat\beta = \arg\min_{\beta \in \mathcal{O}_L}\sum_{i=1}^L \mathcal{L}_{\text{match}}(\hat{\mathcal{Y}}_{\beta(i)}, \mathcal{Y}'_i), \qquad \mathcal{L}_{\text{match}} = -\log p^c(t_{\beta(i)}) - \mathbb{I}_{t_{\beta(i)}\neq\varnothing}\log p_i^n(s_{\beta(i)}, e_{\beta(i)}).$$

**Results.** GENIA (nested): F1 = 80.74 (SOTA at submission). CoNLL 2003 (flat): F1 = 93.00. WeiboNER (Chinese): F1 = 72.38. Per-length results show performance is highest for single-token entities (GENIA F1 = 83.41) and degrades with span length due to the pyramid kernel-size constraint on proposal expected length.

---

## 6. Connecting Threads

**Shared structure.** All four papers address the same core problem — scoring or refining candidate spans — but differ in how they enumerate the candidate space and what learning signal non-entity spans receive. Yu et al. (2020) score every $(i,j)$ pair via a biaffine tensor and classify each into an entity type or null; Li (2021) filters the candidate space in stage 1 to a small high-recall subset before classification; Tang et al. (2023) keep the full biaffine tensor but replace null labels with signed offset targets; Wen et al. (2023) abandon explicit enumeration entirely and instead generate a fixed set of learned proposals that are iteratively refined until they converge on entity boundaries.

**Supervision signal design.** The key axis of variation is what non-entity spans or proposals receive as supervision. Yu et al. assign a null category — a pure negative signal. Li (2021) sidesteps most non-entities by filtering in stage 1, then uses hard-negative augmentation to sharpen the remaining negatives. Tang et al. (2023) assign a positive regression target (offset distance) to every non-entity span, eliminating pure negatives entirely. Wen et al. (2023) adopt Hungarian matching: each proposal is assigned either a ground-truth entity or a **None** target, so the model optimizes the joint type-and-location matching cost even for unmatched proposals.

**Proposal initialization vs. enumeration.** Li (2021) and Yu et al. (2020) still enumerate spans explicitly (all pairs up to a length limit, or all $N^2$ pairs). Tang et al. (2023) enumerate the full $N^2$ grid but annotate each cell with a structured label rather than binary. Wen et al. (2023) break from enumeration: the feature pyramid generates $L$ proposals whose initial boundaries come from learned aggregation over multi-scale representations, and the regressor refines these boundaries continuously — closer in spirit to bounding-box regression in object detection than to span classification.

**Nested NER.** The biaffine formulation (Yu et al.) handles nesting by allowing contained span pairs in greedy selection. Li (2021) achieves the same via independent per-proposal prediction. BOPN handles nesting via parallel prediction over all entity types in the 3D offset tensor. Wen et al. (2023) handle nesting naturally: multiple proposals can converge to overlapping spans because they are trained and decoded independently.

**Data efficiency.** Li (2021) and Tang et al. (2023) both report improved low-data performance. Wen et al. (2023) show strong results on WeiboNER, which is a relatively small Chinese social media dataset, attributing this to the spatial attention mechanism's locality bias. The pyramid proposer also reduces the effective search space by initializing proposals near plausible entity locations rather than from scratch.
