# NER: Span Detection, Distant Supervision, and Cross-Domain Transfer

**Topic:** Moving beyond per-token sequence labeling toward span-centric formulations; training NER from noisy distant supervision without manual annotation; and evaluating cross-domain generalization.

**Papers covered:** 5 papers — AutoNER, NER-as-Dependency-Parsing, MRC-for-NER, BOND, CrossNER — plus the BiLSTM-CRF baseline referenced throughout.

---

## 1. The General Problem

Named entity recognition assigns to each token $x_t$ in a sequence $\mathbf{x} = (x_1,\ldots,x_T)$ a label from $\mathcal{Y} \cup \{O\}$, where $\mathcal{Y}$ is a set of entity types and $O$ denotes non-entity. The standard supervised formulation trains a model by minimizing

$$\widehat{\theta} = \arg\min_\theta \frac{1}{M}\sum_{m=1}^M \ell(\mathbf{Y}_m, f(\mathbf{X}_m;\theta))$$

over $M$ manually annotated sentence–label pairs, where $\ell$ is cross-entropy. The canonical baseline (`neural_architectures_named_entity_recognition`) encodes each token with a BiLSTM over character and word embeddings and decodes with a linear-chain CRF:

$$p(\mathbf{y}\mid\mathbf{x}) = \frac{\exp\!\bigl(\sum_t (\mathbf{W}_{y_t}^\top \mathbf{h}_t + T_{y_{t-1},y_t})\bigr)}{\sum_{\mathbf{y}'}\exp\!\bigl(\sum_t (\mathbf{W}_{y'_t}^\top \mathbf{h}_t + T_{y'_{t-1},y'_t})\bigr)},$$

where $T \in \mathbb{R}^{|\mathcal{Y}|\times|\mathcal{Y}|}$ is a learned transition matrix and Viterbi decoding recovers the MAP sequence. This formulation has two structural limitations: (1) it assigns a single label per token, incompatible with nested entities; and (2) it never scores a span as a unit — entity boundaries are encoded only implicitly through the BIO prefix system. The papers below address these limitations through span-level reformulations, and separately address the data requirement through distant supervision.

---

## 2. Span-Based Reformulations

### 2.1 NER as Dependency Parsing — Yu et al., ACL 2020 · `named_entity_recognition_dependency_parsing`

**Core idea.** Reformulate NER as scoring all possible (start, end) index pairs via a biaffine model, giving the system a global view of entity boundaries.

**Encoding.** Input tokens are encoded through a multi-layer BiLSTM over BERT$_\text{Large}$, fastText, and character CNN embeddings, producing contextual representations $\mathbf{h}_1, \ldots, \mathbf{h}_T$. Two separate FFNNs produce start and end representations per token position:

$$h_s(i) = \text{FFNN}_s(\mathbf{h}_{s_i}), \quad h_e(i) = \text{FFNN}_e(\mathbf{h}_{e_i}).$$

Using separate networks reflects the observation that the head and tail tokens of an entity span play structurally different syntactic roles.

**Span scoring.** For each NER category $m$ (including non-entity), the score of span $i$ with start $s_i$ and end $e_i$ is:

$$r_m(i) = h_s(i)^\top \mathbf{U}_m\, h_e(i) + W_m\bigl(h_s(i) \oplus h_e(i)\bigr) + b_m,$$

where $\mathbf{U}_m \in \mathbb{R}^{d \times c \times d}$ is a biaffine tensor, $W_m \in \mathbb{R}^{c \times 2d}$ and $b_m$ are affine parameters, and $c$ is the number of entity categories plus one. This yields an $l \times l \times c$ scoring tensor over all candidate spans.

**Training.** Softmax cross-entropy over all valid spans:

$$\text{loss} = -\sum_{i=1}^N \sum_{c=1}^C y_{i_c} \log p_m(i_c), \quad p_m(i_c) = \frac{\exp(r_{i_c})}{\sum_{c'} \exp(r_{i_{c'}})}.$$

**Decoding.** Candidate spans are ranked by their predicted category score and greedily selected subject to boundary constraints. For flat NER, no two spans may overlap. For nested NER, spans may contain each other but not partially cross: span $i$ clashes with span $j$ if $s_i < s_j \leq e_i < e_j$.

**Results** (F1 on flat NER benchmarks):

| Benchmark | Previous SoTA | This model | $\Delta$ |
|---|---|---|---|
| CoNLL 2003 English | 93.5 | **93.7** | +0.2 |
| OntoNotes | 89.0 | **91.2** | +2.2 |
| CoNLL 2002 Spanish | 88.8 | **90.3** | +1.5 |
| ACE 2004 (nested) | 84.7 | **87.3** | +2.6 |

---

### 2.2 A Unified MRC Framework for NER — Li et al., ACL 2020 · `unified_mrc_framework_named_entity_recognition`

**Core idea.** Cast each entity type as a natural language question; extract entity spans as MRC answer spans. This eliminates the single-label-per-token constraint for nested NER without any architectural change.

**Problem reformulation.** Each entity type $y \in Y$ is associated with a query $q_y$ derived from annotation guidelines (e.g., for PER: *"which person is mentioned in the text?"*). An annotated entity span $x_{\text{start,end}}$ becomes an answer to $q_y$ in context $X$, yielding a (QUESTION, ANSWER, CONTEXT) triple $(q_y,\, x_{\text{start,end}},\, X)$.

**Model.** The concatenation $[\text{[CLS]}\; q_y\; \text{[SEP]}\; X\; \text{[SEP]}]$ is fed to BERT. Two linear heads over the output representations predict start and end position distributions:

$$p^\text{start} = \text{softmax}(\mathbf{W}^\text{start} \mathbf{H}^\top), \quad p^\text{end} = \text{softmax}(\mathbf{W}^\text{end} \mathbf{H}^\top),$$

where $\mathbf{H} \in \mathbb{R}^{n \times d}$ is the BERT output. Training minimizes cross-entropy jointly over start and end positions for each entity type.

**Why MRC handles nesting.** Extracting two overlapping entities with categories $k$ and $k'$ requires answering queries $q_k$ and $q_{k'}$ independently. Each extraction is a separate forward pass; there is no representational conflict. The sequence labeling formulation cannot assign two BIO labels to the same token, while the MRC formulation has no such constraint.

**Semantic prior.** The query encodes knowledge about the entity category semantically. The query for ORG (*"find a company, agency or institution in the context"*) guides BERT's cross-attention toward organizationally-relevant tokens, a richer signal than the one-hot ORG label used in sequence labeling.

**Results** (F1 improvements over prior SoTA):

| Benchmark | $\Delta$F1 |
|---|---|
| ACE04 (nested) | +1.28 |
| ACE05 (nested) | +2.55 |
| GENIA (nested) | +5.44 |
| KBP17 (nested) | +6.37 |
| CoNLL 2003 (flat) | +1.49 |
| OntoNotes 5.0 (flat) | +0.21 |

---

## 3. Distant Supervision: Replacing Manual Annotation

Both methods below train NER without any manually annotated sentences, instead generating labels automatically from external resources (dictionaries or knowledge bases).

### 3.1 AutoNER — Shang et al., EMNLP 2018 · `learning_named_entity_tagger_domainspecific_dictionary`

**Setting.** Labels are generated by matching text against entity dictionaries. This produces two failure modes: tokens absent from the dictionary yield false-negative $O$ labels; and the exact entity boundaries may be mis-specified.

**Fuzzy-LSTM-CRF baseline.** To handle multi-label tokens, the conventional CRF objective $p(\mathbf{y}|X)$ is replaced by a fuzzy CRF that maximizes total probability over all valid label sequences under the modified IOBES scheme:

$$p(y|X) = \frac{\sum_{\tilde{y} \in Y_{\text{possible}}} e^{s(X,\tilde{y})}}{\sum_{\hat{y} \in Y_X} e^{s(X,\hat{y})}},$$

where $Y_{\text{possible}}$ is the set of valid sequences given dictionary constraints and $s(X, y) = \sum_{i=0}^n \Phi_{y_i,y_{i+1}} + \sum_{i=1}^n P_{i,y_i}$ is the standard CRF score with transition matrix $\Phi$ and emission scores $P_{i,y_i}$. When all labels are unique and known, this reduces exactly to the conventional CRF.

**AutoNER with Tie-or-Break tagging.** Instead of per-token IOBES labels, AutoNER predicts whether adjacent tokens are tied in the same entity mention (Tie) or broken into separate chunks (Break), with Unknown for tokens inside out-of-dictionary high-quality phrases. A BiLSTM encodes the sequence and a sigmoid binary classifier predicts:

$$p(y_i = \text{Break}\mid\mathbf{u}_i) = \sigma(\mathbf{w}^\top \mathbf{u}_i).$$

The span detection loss skips Unknown positions:

$$\mathcal{L}_\text{span} = \sum_{i \mid y_i \neq \text{Unknown}} \ell\!\bigl(y_i,\; p(y_i = \text{Break}\mid\mathbf{u}_i)\bigr).$$

Tokens between consecutive Break tags form a candidate entity span. A second-stage softmax classifier over span representation $\mathbf{v}_i$ assigns entity types using soft supervision — since each span may match multiple dictionary types:

$$\mathcal{L}_\text{type} = H\!\bigl(\hat{p}(\cdot\mid\mathbf{v}_i, L_i),\; p(\cdot\mid\mathbf{v}_i)\bigr),$$

where $L_i$ is the set of all possible types for span $i$ under dictionary matching and $H$ is cross-entropy with the model distribution $p(\cdot|\mathbf{v}_i) = \text{softmax}(\mathbf{W}^\text{type}\mathbf{v}_i)$.

**Key insight.** Even when dictionary matching mis-specifies entity boundaries, the internal ties between tokens within the span are often correct. The Tie-or-Break scheme is therefore robust to boundary noise in the distant labels.

**Results** (F1 vs. fully supervised BiLSTM-CRF, no manual annotation):

| Dataset | Fully supervised | AutoNER | Gap |
|---|---|---|---|
| BC5CDR | 88.6 | 85.3 | −3.3 |
| NCBI-disease | 87.0 | 79.6 | −7.4 |
| LaptopReview | 80.1 | 73.3 | −6.8 |

---

### 3.2 BOND — Liang et al., KDD 2020 · `bond_bertenhanced_opendomain_named_entity_recognition`

**Setting.** Open-domain NER with distant labels generated by matching text to large knowledge bases (Wikipedia, YAGO). Two challenges compound: incomplete annotation (entity tokens labeled $O$ due to low KB coverage — below 60% on standard benchmarks) and noisy annotation (the same surface form mapped to multiple entity types).

**Stage 1: Early-stopped fine-tuning.** RoBERTa is fine-tuned on distantly-matched labels minimizing:

$$\widehat{\theta} = \arg\min_\theta \frac{1}{M}\sum_{m=1}^M \ell(\mathbf{Y}_m^\text{distant}, f(\mathbf{X}_m;\theta)),$$

where $\ell$ is cross-entropy. Training is terminated early before convergence: since incomplete annotation biases the labels toward $O$, continued training would overfit to predicting non-entity. Early stopping lets RoBERTa's pre-trained entity semantics dominate rather than the noisy $O$ signal. After stage 1, the model generates pseudo soft-label distributions $\tilde{p}(y\mid x;\theta_T)$ for all tokens.

**Stage 2: Teacher–student self-training.** A student model $\theta_S$ is initialized from the stage-1 checkpoint and trained on pseudo soft-labels from a teacher model $\theta_T$. The student loss is:

$$\mathcal{L}_\text{student} = -\sum_{x \in \mathcal{D}_\text{conf}} \sum_y \tilde{p}(y\mid x;\theta_T) \log p(y\mid x;\theta_S),$$

where $\mathcal{D}_\text{conf}$ is the subset of tokens whose teacher prediction exceeds a confidence threshold (selected by prediction entropy or top-$k$ confidence). The teacher is updated from the student at each iteration, progressively improving pseudo-label quality as the student becomes more accurate. Roles alternate: teacher generates pseudo-labels, student trains on them, teacher is then updated from student.

**Results** (F1, distantly-supervised NER, improvement over prior best):

| Dataset | Prior best | BOND | $\Delta$ |
|---|---|---|---|
| CoNLL03 | 76.0 | **81.5** | +5.5 |
| Tweet | 45.5 | **57.1** | +11.6 |
| OntoNotes | 63.8 | **65.2** | +1.4 |

---

## 4. Cross-Domain Evaluation

### 4.1 CrossNER — Liu et al., AAAI 2021 · `crossner_evaluating_crossdomain_named_entity_recognition`

**Dataset.** CrossNER provides five target domains — politics, natural science, music, literature, AI — each with domain-specialized entity categories not present in the CoNLL-2003 news source domain (e.g., "politician", "election", "political party" in politics; "algorithm", "researcher", "task" in AI). Training sets are small by design: 100–200 labeled sentences per domain, reflecting the low-resource cross-domain transfer setting. Each domain is paired with a large unlabeled domain corpus (1.72M–9.82M sentences) sourced from Wikipedia for domain-adaptive pre-training.

**Domain-adaptive pre-training (DAPT).** BERT is continued pre-training on the unlabeled domain corpus before NER fine-tuning. The paper investigates two axes.

*Corpus selection.* Using the full domain corpus vs. a fractional subset emphasizing sentences containing domain-specialized entities. The fractional corpus consistently outperforms the full corpus: domain-irrelevant sentences add noise without improving entity-specific representations.

*Masking strategy.* Standard random-token MLM vs. Entity-Span-MLM, which masks named entity tokens with higher probability. Formally, if $e_t \in \{0,1\}$ indicates whether token $t$ is a named entity, Entity-Span-MLM samples each token for masking with probability:

$$p_\text{mask}(t) = \lambda_e\, e_t + \lambda_O\,(1 - e_t), \quad \lambda_e \gg \lambda_O.$$

This forces the model to learn entity-specific contextual representations rather than general language statistics.

**Domain gap analysis.** Vocabulary overlap between source (Reuters) and each target domain is measured as the fraction of top-5K frequent words (excluding stopwords) shared between domains. All pairwise overlaps are below 35.7%, confirming genuine domain shift rather than merely surface-level vocabulary differences.

**Results** (F1, BERT fine-tuned on labeled target data, CoNLL source):

| Target domain | No DAPT | DAPT (full corpus, random MLM) | DAPT (entity corpus, Entity-Span-MLM) |
|---|---|---|---|
| Politics | 70.2 | 73.8 | **76.9** |
| Natural Science | 64.5 | 67.2 | **70.1** |
| Music | 55.3 | 59.1 | **63.4** |
| Literature | 59.0 | 62.0 | **65.7** |
| AI | 48.1 | 51.4 | **56.3** |

---

## 5. Connecting Threads

**The detection–typing decomposition recurs independently.** AutoNER separates span detection (Tie-or-Break classifier) from type assignment (softmax over span representation). The biaffine model separates start/end scoring (biaffine tensor $\mathbf{U}_m$) from category assignment (argmax over the $c$ scores). The MRC model separates span extraction (start/end position heads) from type assignment (separate query per type). This convergence from three different motivations — distant supervision noise, nested entity handling, and zero-shot type transfer — suggests that the detection-then-typing structure is a natural inductive bias for NER rather than an architectural convenience.

**The O-class is structurally different from entity classes.** In the distant supervision setting (AutoNER, BOND), $O$ tokens dominate numerically and their labels are systematically unreliable (false negatives). In the span-based setting (biaffine, MRC), $O$ spans are any (start, end) pair that does not constitute a named entity — a semantically heterogeneous and unbounded set. Both lines of work handle $O$ through structural choices rather than metric similarity: BOND uses early stopping to avoid $O$-bias; AutoNER skips Unknown tokens from the span detection loss; the biaffine model includes a non-entity category in $r_m$ but ranks entity-class spans separately; MRC only extracts spans above a confidence threshold, implicitly treating everything else as $O$.

**Pre-trained representations absorb noisy distant labels.** BOND's central finding is that RoBERTa's pre-trained entity semantics survive early-stopped fine-tuning on noisy distant labels, providing a reliable initialization for the teacher-student stage. CrossNER's finding is analogous: domain-adaptive BERT pre-training on entity-bearing text transfers entity representations that fine-tuning on 100–200 labeled sentences alone cannot achieve. In both cases, the pre-trained model carries an entity-world prior that reduces the effective supervision requirement.

**Span scoring gives a global sentence view; BIO labeling does not.** In the BiLSTM-CRF, the model at position $t$ sees only a left-to-right (and right-to-left) summary of the context via the hidden states. The biaffine span scorer simultaneously considers all $(s_i, e_i)$ pairs, letting the score of a long span depend jointly on the representations of its boundary tokens. This global view is critical for nested NER (where the same token participates in spans at multiple granularities) and for long entities (where the boundary tokens are distant in the sequence).
