# Entity Linking and Coreference Resolution

**Topic:** Identifying and grounding entity mentions in text — either by clustering spans that refer to the same real-world entity (coreference resolution) or by anchoring each span to an entry in a structured knowledge base (entity linking).

**Papers covered:** 7 papers spanning the canonical span-ranking coreference architecture, its higher-order extension, the dominant span pre-training method, three entity linking paradigms (cross-encoder, bi-encoder + re-ranker, autoregressive), and a comprehensive survey of the field.

---

## 1. The Problem: Two Related Tasks

### 1.1 Coreference Resolution

Given a document $D$ with $T$ tokens, there are $N = T(T+1)/2$ candidate text spans. Each span is indexed by its start and end positions, $(\text{START}(i), \text{END}(i))$. The task is to partition all entity-denoting spans into **coreference clusters** — groups that refer to the same real-world entity.

The canonical formulation (Lee et al., 2017) converts this into a per-span antecedent assignment problem. Each span $i$ is assigned an antecedent:

$$y_i \in \mathcal{Y}(i) = \{\epsilon,\, 1,\, \ldots,\, i-1\}$$

where $\epsilon$ is a dummy antecedent meaning "this span starts a new cluster or is not a mention at all." All spans connected by a chain of antecedent predictions are recovered as a single cluster. Crucially, this formulation is **unlabelled**: the clusters carry no entity-type information, only identity.

The standard evaluation benchmark is the English OntoNotes dataset (CoNLL-2012 shared task), evaluated with the average F1 of three metrics: MUC, $\text{B}^3$, and CEAF$_{\phi_4}$.

### 1.2 Entity Linking

Given a document $D$, an entity linking (EL) system must:

1. **Mention Detection (MD):** identify the text spans $m_1, \ldots, m_n$ that are entity mentions.
2. **Entity Disambiguation (ED):** link each mention $m_i$ to an entry $e_i \in \mathcal{E}$ in a knowledge base (KB), or predict NIL if no entry matches.

Formally, with contexts $C$ and entity set $E$:

$$\text{MD} : C \to M^n, \qquad \text{ED} : (M, C)^n \to (E \cup \{\text{NIL}\})^n$$

Most work assumes mention boundaries are given and focuses on ED. The end-to-end setting solves both jointly:

$$\text{EL} : C \to (M, E)^n$$

Entities in $\mathcal{E}$ are typically identified by textual descriptions $d_e$ (e.g., Wikipedia article intros). The core challenge is **disambiguation**: the mention "Jaguar" may refer to the car brand, the animal, or the operating system depending on context, and the system must select the correct KB entry.

**Zero-shot entity linking** (Logeswaran et al., 2019) adds a harder constraint: $\mathcal{E}_\text{train} \cap \mathcal{E}_\text{test} = \emptyset$. The KB entities seen at test time are entirely unseen during training. No alias tables, frequency statistics, or structured metadata are allowed — the model must generalise purely through language understanding, matching mention context against entity descriptions.

### 1.3 Relationship Between the Two Tasks

Coreference and EL are complementary. Coreference groups mentions that are co-referent without knowing *what* entity they refer to. EL identifies *which* KB entity a mention refers to without necessarily knowing which other spans in the document refer to the same entity. A complete pipeline may chain them: (1) resolve coreference clusters, (2) link one representative mention per cluster to the KB, propagating the assignment to the full cluster.

---

## 2. Span Representations: The Shared Foundation

All modern coreference and many EL systems represent entity mentions as **span embeddings** — fixed-length vectors computed from the tokens within and around a candidate span. Understanding span representation is therefore a prerequisite for all subsequent sections.

### 2.1 BiLSTM Span Representations — Lee et al., 2017 · `endtoend_neural_coreference_resolution`

The simplest span representation combines four signals. For a span $i$ covering token positions $[\text{START}(i), \text{END}(i)]$:

$$\boldsymbol{g}_i = \bigl[\boldsymbol{x}^*_{\text{START}(i)},\; \boldsymbol{x}^*_{\text{END}(i)},\; \hat{\boldsymbol{x}}_i,\; \phi(i)\bigr]$$

- $\boldsymbol{x}^*_t = [\overrightarrow{\boldsymbol{h}}_t;\, \overleftarrow{\boldsymbol{h}}_t]$ are the concatenated forward/backward BiLSTM hidden states, providing sentential context at each boundary.
- $\hat{\boldsymbol{x}}_i = \sum_{t=\text{START}(i)}^{\text{END}(i)} a_{i,t}\, \boldsymbol{x}_t$ is a soft head-word vector, where the attention weights $a_{i,t} \propto \exp(\boldsymbol{w}_\alpha^\top \text{FFNN}_\alpha(\boldsymbol{x}^*_t))$ are learned from supervision on coreference clusters alone (no syntactic parse).
- $\phi(i)$ is a learned embedding of the span width.

This representation encodes (i) the global sentence context at span boundaries, (ii) the most salient internal token, and (iii) a length prior. All subsequent coreference models, including the higher-order extension and SpanBERT, refine this basic template.

### 2.2 SpanBERT — Joshi et al., TACL 2020 · `spanbert_improving_pretraining_representing_predicting_spans`

SpanBERT replaces the BiLSTM with a BERT encoder and introduces two span-specific modifications to the standard BERT pre-training objective. Standard BERT pre-training randomly masks ~15% of individual tokens and trains the encoder to predict each one from surrounding context (Masked Language Modelling, MLM). SpanBERT replaces this with:

**Span masking.** Rather than masking independently sampled tokens, contiguous word spans are masked. Span lengths are drawn from a geometric distribution $\ell \sim \text{Geo}(p = 0.2)$, clipped at $\ell_\text{max} = 10$ (mean $\approx 3.8$ words). This forces the encoder to model within-span lexical coherence rather than exploiting immediately adjacent visible tokens.

**Span Boundary Objective (SBO).** For a masked span $(x_s, \ldots, x_e)$, each masked token $x_i$ must be predicted from the **boundary** representations $\boldsymbol{x}_{s-1}$ and $\boldsymbol{x}_{e+1}$ and a relative position embedding $\boldsymbol{p}_{i-s+1}$:

$$\boldsymbol{y}_i = f(\boldsymbol{x}_{s-1},\, \boldsymbol{x}_{e+1},\, \boldsymbol{p}_{i-s+1})$$

where $f$ is a two-layer MLP with GeLU activations and LayerNorm. The total per-token loss sums the standard MLM loss and the SBO loss:

$$\mathcal{L}(x_i) = \mathcal{L}_\text{MLM}(x_i) + \mathcal{L}_\text{SBO}(x_i)$$

The SBO objective forces boundary token representations to encode the semantic content of the span they delimit — precisely the information needed by downstream span-selection models that represent spans as a function of their boundary encodings. SpanBERT also drops the Next Sentence Prediction objective and trains on single contiguous segments up to 512 tokens.

**Results.** SpanBERT achieves 79.6 avg. F1 on the OntoNotes coreference task (+6.6 over prior best), 88.8/94.6 EM/F1 on SQuAD 1.1, and +3.3 F1 on TACRED relation extraction, demonstrating that span-aware pre-training transfers broadly to span-selection tasks.

---

## 3. Coreference Resolution

### 3.1 End-to-End Span Ranking — Lee et al., EMNLP 2017 · `endtoend_neural_coreference_resolution`

**Core idea.** Previous coreference systems required hand-engineered mention detectors and syntactic parsers as preprocessing. This paper eliminates both by treating all $N$ candidate spans simultaneously and jointly learning which spans are mentions and which pairs are coreferent.

**Scoring.** The full joint distribution is factored span-by-span, each following a softmax over antecedents:

$$P(y_i \mid D) = \frac{\exp(s(i,\, y_i))}{\sum_{y' \in \mathcal{Y}(i)} \exp(s(i,\, y'))}$$

The pairwise coreference score decomposes into three factors:

$$s(i, j) = \begin{cases} 0 & j = \epsilon \\ s_m(i) + s_m(j) + s_a(i, j) & j \neq \epsilon \end{cases}$$

- $s_m(i) = \boldsymbol{w}_m \cdot \text{FFNN}_m(\boldsymbol{g}_i)$ is the unary mention score: how likely is span $i$ to be an entity mention at all?
- $s_a(i,j) = \boldsymbol{w}_a \cdot \text{FFNN}_a([\boldsymbol{g}_i,\, \boldsymbol{g}_j,\, \boldsymbol{g}_i \circ \boldsymbol{g}_j,\, \phi(i,j)])$ is the antecedent score, where $\circ$ is element-wise product and $\phi(i,j)$ encodes speaker identity and span distance.

Fixing the dummy score $s(i, \epsilon) = 0$ means: the model predicts a coreferent antecedent if and only if some $s(i,j) > 0$.

**Pruning.** Scoring all $O(T^4)$ span pairs naively is intractable. The factored design enables two-stage pruning: (1) keep the top $\lambda T$ spans by mention score $s_m(i)$ alone; (2) for surviving spans, consider at most $K$ previous spans as antecedents. At $\lambda = 0.4$ and $K = 250$, over 92% of gold mentions are retained.

**Training.** Only coreference cluster assignments (not individual antecedent links) are observed. The model maximises the marginal log-likelihood over all correct antecedents implied by each gold cluster:

$$\mathcal{L} = \log \prod_{i=1}^N \sum_{\hat{y} \in \mathcal{Y}(i) \cap \text{GOLD}(i)} P(\hat{y} \mid D)$$

where $\text{GOLD}(i)$ is the set of spans in the same gold cluster as span $i$, or $\{\epsilon\}$ if span $i$ is not in any cluster.

**Results.** Single model: **67.2 avg. F1** on OntoNotes, improving the prior state of the art by +1.5 F1; 5-model ensemble: **68.8 avg. F1** (+3.1). The most important features are span distance/width embeddings (−3.8 F1 when ablated) and the soft head-finding attention (−1.3 F1).

---

### 3.2 Higher-Order Inference — Lee et al., EMNLP 2018 · `higherorder_coreference_resolution_coarsetofine_inference`

**The first-order failure mode.** The Lee et al. (2017) model scores only pairs of spans. Each antecedent decision is made independently, making globally inconsistent clusters possible. Consider three spans $[I]$, $[you]$, $[all\ of\ you]$: the pair $(I, you)$ may be locally compatible, and $(you, all\ of\ you)$ may be locally compatible, but the triplet $\{I, you, all\ of\ you\}$ is inconsistent in plurality. A first-order model has no mechanism to rule this out.

**Higher-order inference via iterative refinement.** At each iteration $n$, the antecedent distribution $P_n(y_i)$ is used as an attention mechanism to compute an expected antecedent representation:

$$\boldsymbol{a}^n_i = \sum_{y_i \in \mathcal{Y}(i)} P_n(y_i) \cdot \boldsymbol{g}^n_{y_i}$$

The span representation is then updated via a learned gate:

$$\boldsymbol{f}^n_i = \sigma\!\bigl(\mathbf{W}_f [\boldsymbol{g}^n_i,\, \boldsymbol{a}^n_i]\bigr), \qquad \boldsymbol{g}^{n+1}_i = \boldsymbol{f}^n_i \circ \boldsymbol{g}^n_i + (1 - \boldsymbol{f}^n_i) \circ \boldsymbol{a}^n_i$$

The gate $\boldsymbol{f}^n_i \in (0,1)^d$ controls, dimension-by-dimension, whether to retain the current span's own information or blend in its expected antecedent's information. After $N$ iterations, span $i$'s representation softly encodes information from up to $N$ spans in its predicted cluster — effectively modelling chains of any length. Empirically, $N=2$ (second order) gives the best trade-off; $N=3$ adds only 0.1 F1.

**Coarse-to-fine antecedent pruning.** The expensive antecedent score $s_a(i,j)$ — which concatenates $\boldsymbol{g}_i$, $\boldsymbol{g}_j$, $\boldsymbol{g}_i \circ \boldsymbol{g}_j$, and $\phi(i,j)$ — must be recomputed at every iteration, and must be evaluated for all $M \times K$ surviving antecedent pairs. To reduce $K$ aggressively, a cheap bilinear coarse score is introduced:

$$s_c(i, j) = \boldsymbol{g}_i^\top \mathbf{W}_c\, \boldsymbol{g}_j$$

where $\mathbf{W}_c \in \mathbb{R}^{d \times d}$ is learned. The final pairwise score becomes:

$$s(i, j) = s_m(i) + s_m(j) + s_c(i, j) + s_a(i, j)$$

and a three-stage beam search is applied: (1) keep top $M$ spans by $s_m$; (2) keep top $K$ antecedents per span by $s_m(i) + s_m(j) + s_c(i,j)$; (3) compute $s_a$ only for surviving pairs. The coarse factor $s_c$ can be computed for all antecedents in $O(M \times M)$ matrix multiplications, making the model nearly insensitive to reducing $K$ from 250 to 50 (less than 0.2 F1 drop vs. nearly 5 F1 with heuristic distance-based pruning).

**Results.** Full model (second-order + coarse-to-fine + ELMo): **73.0 avg. F1** on OntoNotes, a new state of the art at the time and +5.8 over the first-order baseline matched on the same training setup.

---

## 4. Entity Linking

### 4.1 Zero-Shot EL by Reading Descriptions — Logeswaran et al., ACL 2019 · `zeroshot_entity_linking_reading_entity_descriptions`

**Task definition.** The target entity dictionary $\mathcal{E} = \{(e_i, d_i)\}_{i=1}^K$ consists of entity-description pairs. At test time, $\mathcal{E}_\text{test}$ is entirely disjoint from $\mathcal{E}_\text{train}$. No alias tables, link-frequency statistics, or structured data are available. The model must rank candidates using only textual descriptions and mention context.

**Architecture comparison.** The paper evaluates three architectures:

*Pool-Transformer (bi-encoder).* Two separate BERT encoders independently produce single-vector representations: $\boldsymbol{h}_m = \text{BERT}_1([m;\text{ctx}])_{[\text{CLS}]}$ and $\boldsymbol{h}_e = \text{BERT}_2([d_e])_{[\text{CLS}]}$. Score: $w^\top [\boldsymbol{h}_m; \boldsymbol{h}_e]$. Fast at inference (entity embeddings are pre-computable), but cross-attention between mention and description is blocked.

*Full-Transformer (cross-encoder).* Mention context and entity description are concatenated into a single input $[\text{CLS}]\, m\, [\text{SEP}]\, d_e\, [\text{SEP}]$ processed by one BERT model. The $[\text{CLS}]$ representation $\boldsymbol{h}_{m,e}$ is projected to a scalar: $w^\top \boldsymbol{h}_{m,e}$. This allows full bidirectional cross-attention between every mention token and every description token at every layer. Ablations show it outperforms Pool-Transformer by 8–23 accuracy points, establishing that cross-attention is essential when no external cues are available.

**Domain-Adaptive Pre-training (DAP).** Domain shift between training and test worlds causes a performance drop of ~11 points. The remedy is a penultimate pre-training stage on the unlabelled target-domain text using masked language modelling:

$$U_\text{WB} \;\to\; U_\text{src+tgt} \;\to\; U_\text{tgt} \;\to\; \text{fine-tune on source labels}$$

Each arrow is a pre-training stage. The target-domain MLM accuracy after DAP correlates directly with downstream EL accuracy (Figure 2(b) of the paper), showing that better language understanding of the target domain drives better mention-entity matching.

**Results.** Best pipeline achieves **77.05% normalised accuracy** on held-out test domains. Replacing the description-matching model with edit-distance or TF-IDF baselines gives only ~26%, establishing that deep language understanding of descriptions is non-trivially required.

---

### 4.2 BLINK — Wu et al., EMNLP 2020 · `scalable_zeroshot_entity_linking_dense_entity`

**Core idea.** The cross-encoder of Logeswaran et al. is accurate but cannot scale: scoring each of 5.9 million Wikipedia entities for every mention is infeasible. BLINK separates the problem into a scalable retrieval stage (bi-encoder) followed by a precise re-ranking stage (cross-encoder).

**Bi-encoder.** Two independent BERT encoders map mention context and entity description to vectors:

$$\boldsymbol{y}_m = \text{red}(T_1(\tau_m)), \qquad \boldsymbol{y}_e = \text{red}(T_2(\tau_e))$$

where $\text{red}(\cdot)$ extracts the $[\text{CLS}]$ token output. The mention context is formatted as:

$$[\text{CLS}]\; \text{ctx}_l\; [\text{M}_s]\; \text{mention}\; [\text{M}_e]\; \text{ctx}_r\; [\text{SEP}]$$

and the entity as $[\text{CLS}]\; \text{title}\; [\text{ENT}]\; \text{description}\; [\text{SEP}]$. The score for candidate $e_i$ is the dot product $s(m, e_i) = \boldsymbol{y}_m \cdot \boldsymbol{y}_{e_i}$. Training maximises the correct entity's score against in-batch negatives (plus mined hard negatives):

$$\mathcal{L}(m_i, e_i) = -s(m_i, e_i) + \log \sum_{j=1}^{B} \exp(s(m_i, e_j))$$

Because $\boldsymbol{y}_e$ does not depend on the query, all entity embeddings can be pre-computed once and stored. Retrieval reduces to a maximum inner-product search over a FAISS index, linking against 5.9M Wikipedia entities in under 2 ms per query.

**Cross-encoder.** The top $k$ candidates from the bi-encoder are re-ranked by a single BERT that jointly encodes mention and entity: $\tau_{m,e} = [\text{CLS}]\; \text{ctx}_l\; m\; \text{ctx}_r\; [\text{SEP}]\; d_e\; [\text{SEP}]$. A linear layer on the $[\text{CLS}]$ embedding scores the pair:

$$s_\text{cross}(m, e) = \boldsymbol{y}_{m,e}\, \mathbf{W}$$

**Knowledge distillation.** The accuracy gap between the fast bi-encoder and the accurate cross-encoder can be narrowed by distilling the cross-encoder's soft logits into the bi-encoder. With temperature $\tau$, the distillation target is:

$$\sigma(z, \tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

and the combined student loss is $\mathcal{L} = \alpha\, \mathcal{L}_\text{st} + (1 - \alpha)\, \mathcal{L}_\text{dist}$ where $\mathcal{L}_\text{dist} = \mathcal{H}(\sigma(z_t;\tau),\, \sigma(z_s;\tau))$ is cross-entropy between teacher and student logit distributions.

**Results.** Bi-encoder + cross-encoder: **94.5% accuracy** on TACKBP-2010, exceeding all prior methods that relied on alias tables and entity-type priors. On the zero-shot WikilinksNED Unseen-Mentions benchmark the bi-encoder alone achieves 75.2%, nearly doubling the prior best of 43.4%.

---

### 4.3 GENRE — De Cao et al., ICLR 2021 · `genre_autoregressive_entity_retrieval`

**Rethinking the output space.** Dense retrieval systems (BLINK) require storing a vector for every entity — ~24GB for 6M Wikipedia entities at 1024 dimensions — and must subsample negatives at training time because the exact softmax over all entities is intractable. GENRE sidesteps both problems by treating entity retrieval as **sequence generation**: the model generates the entity's textual name, token by token, conditioned on the input context.

**Scoring.** An entity $e$ identified by name $y = (y_1, \ldots, y_N)$ is scored by an autoregressive model:

$$\text{score}(e \mid x) = p_\theta(y \mid x) = \prod_{i=1}^{N} p_\theta(y_i \mid y_{<i},\, x)$$

The model is BART (a pre-trained encoder-decoder transformer), fine-tuned with a standard teacher-forcing sequence-to-sequence objective. The exact softmax over all vocabulary tokens at each step is feasible (vocabulary size $\ll$ entity count), eliminating the need for negative sampling.

**Constrained beam search.** The model must generate valid entity names — arbitrary generation may produce strings not in $\mathcal{E}$. This is enforced by a **prefix trie** $\mathcal{T}$ over all entity identifiers. At each decoding step, only tokens that correspond to allowed continuations from the current prefix are admitted (all others have their log-probability masked to $-\infty$). For 6M Wikipedia titles, the trie has ~17M internal nodes occupying ~600MB — orders of magnitude less than a dense entity index.

**Memory comparison.** DPR requires 70.9GB (220M parameters + 15B index), RAG requires 40.4GB, BLINK requires 30.1GB, GENRE requires only **2.1GB** (406M parameters + 17M-node trie).

**End-to-end entity linking.** For joint mention detection and linking, GENRE uses a Markup annotation scheme: the model generates the source text augmented with bracket tokens delimiting mention spans and followed by the linked entity name. Since the output space is exponentially large (any markup of the input is potentially valid), the trie is computed dynamically at each decoding step rather than pre-computed.

**Results.** On 20+ datasets across entity disambiguation, end-to-end entity linking, and page-level document retrieval (KILT), GENRE achieves state of the art or near-state of the art in nearly all settings. On KILT retrieval tasks it outperforms the best baseline by +13.7 R-precision points on average. Ablation: replacing textual entity names with numeric IDs drops performance by −20 points on average, showing that the compositional structure of natural-language names is a load-bearing feature of the approach.

---

## 5. Survey: The General EL Architecture — Sevgili et al., Semantic Web Journal 2022 · `neural_entity_linking_survey_models_deep`

This survey distils a generic four-stage pipeline applicable to nearly all neural EL systems published since 2015.

**Stage 1 — Candidate Generation (CG).** Given a mention $m_i$, produce a shortlist of plausible entities:

$$\text{CG} : M \to (e_1, e_2, \ldots, e_k)$$

Three standard approaches: (i) surface-form matching (compare mention string against entity titles and redirects); (ii) alias expansion (expand with synonyms from a pre-built alias table or ontology); (iii) prior probability (rank by $p(e \mid m)$ estimated from hyperlink statistics in Wikipedia). Dense retrieval (as in BLINK) is a fourth approach replacing all three.

**Stage 2 — Context/Mention Encoding.** Produce a dense representation of the mention in its document context:

$$\text{mENC} : (C, M)^n \to (\boldsymbol{y}_{m_1}, \ldots, \boldsymbol{y}_{m_n})$$

Early models used BiLSTMs; current practice uses BERT-based encoders where the mention boundaries are flagged with special tokens and the $[\text{CLS}]$ representation (or a mean-pool of mention tokens) is taken as $\boldsymbol{y}_m$.

**Stage 3 — Entity Encoding.** Produce representations for candidate entities:

$$\text{eENC} : E^k \to (\boldsymbol{y}_{e_1}, \ldots, \boldsymbol{y}_{e_k})$$

Three encoding families: (i) co-occurrence statistics (word2vec trained on entity description pages and surrounding anchor text); (ii) KG graph embeddings (TransE: $\text{head} + \text{relation} \approx \text{tail}$ in embedding space, DeepWalk: random-walk-based); (iii) neural description encoders (BERT encoding of the entity description page, pooled to a single vector).

**Stage 4 — Entity Ranking.** Score each candidate and select:

$$\text{RNK} : ((e_1,\ldots,e_k), C, M)^n \to \mathbb{R}^{n \times k}$$

The standard local similarity measure (dot product or cosine):

$$s(m, e_i) = \boldsymbol{y}_m \cdot \boldsymbol{y}_{e_i}, \qquad P(e_i \mid m) = \frac{\exp(s(m,e_i))}{\sum_{j=1}^k \exp(s(m,e_j))}$$

Training uses either the negative log-likelihood $\mathcal{L}(m) = -s(m, e_*) + \log\sum_i \exp(s(m,e_i))$ or a margin ranking loss $\ell(e_i, m) = [\gamma - \Phi(e_*, m) + \Phi(e_i, m)]_+$ with margin $\gamma > 0$.

**Notable architectural modifications catalogued by the survey:**

- *Joint MD + ED:* Treat EL as sequence labelling (BERT-CRF over spans) or enumerate all $n$-gram spans and filter jointly.
- *Global context:* Disambiguate all mentions in a document simultaneously, enforcing coherence via a CRF potential $\Psi(e_i, m_i, c_i) + \sum_{i<j}\Phi(e_i, e_j)$ maximised approximately with loopy belief propagation or graph neural networks.
- *Domain-independent / zero-shot:* Match mention against entity descriptions with a cross-encoder (as in BLINK, Logeswaran et al.) or generate the entity name autoregressively (GENRE), without relying on alias tables.
- *Cross-lingual:* Share encoder weights across languages using multilingual BERT, or transfer from high-resource to low-resource languages via representation alignment.

---

## 6. Summary Table

| Paper | Year | Task | Core mechanism | Key equation | Primary metric |
|---|---|---|---|---|---|
| Lee et al. | 2017 | Coref | End-to-end span ranking | $s(i,j) = s_m(i) + s_m(j) + s_a(i,j)$ | avg. F1 (OntoNotes) |
| Lee et al. | 2018 | Coref | Higher-order + coarse-to-fine | $\boldsymbol{g}^{n+1}_i = \boldsymbol{f}^n_i \circ \boldsymbol{g}^n_i + (1-\boldsymbol{f}^n_i) \circ \boldsymbol{a}^n_i$ | avg. F1 (OntoNotes) |
| SpanBERT | 2020 | Span repr. | Span masking + SBO pre-training | $\mathcal{L}(x_i) = \mathcal{L}_\text{MLM}(x_i) + \mathcal{L}_\text{SBO}(x_i)$ | F1 (OntoNotes, SQuAD) |
| Logeswaran et al. | 2019 | Zero-shot EL | Cross-encoder + DAP | $s(m,e) = w^\top \boldsymbol{h}_{m,e}$ | norm. accuracy (Wikia) |
| BLINK | 2020 | EL | Bi-encoder retrieval + cross-encoder re-rank | $s(m,e_i) = \boldsymbol{y}_m \cdot \boldsymbol{y}_{e_i}$ | accuracy (TACKBP-2010) |
| GENRE | 2021 | EL | Autoregressive name generation + trie decoding | $\text{score}(e\|x) = \prod_i p_\theta(y_i\|y_{<i}, x)$ | $F_1$ / R-precision (KILT) |
| Sevgili et al. | 2022 | Survey | Four-stage EL pipeline taxonomy | $P(e_i\|m) = \text{softmax}(s(m,e_i))$ | — |

---

## 7. Design Axes

**Mention representation: token-level vs. span-level.** Token-level representations (BERT $[\text{CLS}]$ or per-token mean) are standard in EL and sufficient when mention boundaries are given. Span-level representations (BiLSTM boundary + head attention in Lee et al.; BERT boundary embeddings in SpanBERT's SBO) are necessary when the model must also *detect* entity spans from scratch, as in coreference resolution. SpanBERT's SBO pre-training specifically optimises boundary tokens to summarise span content, making it a better initialisation for span-detection models than vanilla BERT.

**Entity representation: description encoding vs. dense index vs. generative.** Description-encoding bi-encoders (BLINK) are fast at inference (entity embeddings are pre-computed), generalize to new entities by encoding their descriptions, but lose cross-attention precision. Cross-encoders (Logeswaran et al., BLINK reranker) are accurate but cannot scale to millions of candidates; they function as a re-ranking stage over a short-list. Generative retrieval (GENRE) eliminates the dense index entirely, trading off a trie of $O(|\mathcal{E}| \times \bar{L})$ nodes for the entity set, and scales memory with name length rather than entity count.

**First-order vs. higher-order inference.** First-order models (Lee et al., 2017) score span pairs independently, which is sufficient for most coreferent pairs but fails on long chains where transitivity is required. Higher-order inference (Lee et al., 2018) propagates information along predicted clusters via iterative representation refinement, improving precision on chains and reducing globally inconsistent clusters. The coarse-to-fine pruning makes the additional computational cost manageable.

**In-domain vs. zero-shot generalisation.** Standard EL systems exploit alias tables and link-frequency statistics that are specific to a single KB (usually Wikipedia). Zero-shot systems (Logeswaran et al., BLINK, GENRE) assume only textual entity descriptions, enabling transfer to entirely new domains and KBs. The main tools for improving zero-shot performance are: (i) domain-adaptive pre-training on target-domain unlabelled text (Logeswaran et al.); (ii) hard-negative mining during bi-encoder training (BLINK); (iii) exploiting the compositional name structure (GENRE, where generating "Leonardo da Vinci" token-by-token implicitly decomposes the disambiguation into smaller decisions).

**Scalability vs. precision.** The bi-encoder is fast (sub-linear retrieval via FAISS) but loses the cross-attention signal; the cross-encoder is precise but $O(k)$ per mention where $k$ is the candidate set size. The pipeline (bi-encoder retrieval → cross-encoder re-ranking) composes both: the bi-encoder ensures high recall at the top-$k$ shortlist; the cross-encoder ensures high precision from the shortlist. GENRE replaces both with constrained beam search over a prefix trie, scaling with vocabulary size (not entity count) at each decoding step and achieving a memory footprint ~14–34× smaller than dense retrieval systems.

**Pre-training objectives.** Standard BERT pre-training (MLM + NSP) is a generic language model and does not specifically encourage span-level representations. SpanBERT's SBO forces boundary tokens to encode span content, directly improving span-dependent downstream tasks. For EL, BLINK pre-trains encoders on generic BERT then fine-tunes on Wikipedia mention-entity pairs; GENRE pre-trains BART on language modelling then fine-tunes on BLINK's 9M entity triples. There is no universal recipe: the choice of pre-training data and objective should match the downstream task's input-output structure.

---

## 8. Experimental Comparison: Three Architectures

This section maps the three architectures selected for experimental evaluation to their mathematical scoring functions. In all cases, mention boundaries are assumed given; the task is pure entity disambiguation (ED).

**Architecture A — Cross-encoder** · source: Logeswaran et al. (2019), §4.1

Mention context and entity description are concatenated into a single transformer input:

$$\text{input} = [\text{CLS}]\; m \; [\text{SEP}]\; d_e \; [\text{SEP}]$$

Full bidirectional self-attention operates jointly over mention and description tokens at every layer. The $[\text{CLS}]$ embedding $\boldsymbol{h}_{m,e}$ is projected to a scalar score:

$$s_A(m, e) = \boldsymbol{w}^\top \boldsymbol{h}_{m,e}$$

The predicted entity is $\hat{e} = \arg\max_{e \in \mathcal{C}(m)} s_A(m, e)$ over a candidate shortlist $\mathcal{C}(m)$. Training uses cross-entropy over the candidate set. This architecture allows the richest mention–description interaction but requires one forward pass per candidate, so the candidate set must be small (typically $|\mathcal{C}(m)| \leq 64$).

**Architecture B — Bi-encoder retrieval + cross-encoder re-ranking** · source: BLINK, Wu et al. (2020), §4.2

*Stage 1 — retrieval.* Two independent encoders produce single-vector representations:

$$\boldsymbol{y}_m = \text{BERT}_1([\text{CLS}]\; \text{ctx}_l\; [\text{M}_s]\; m\; [\text{M}_e]\; \text{ctx}_r\; [\text{SEP}])_{[\text{CLS}]}$$
$$\boldsymbol{y}_e = \text{BERT}_2([\text{CLS}]\; \text{title}\; [\text{ENT}]\; d_e\; [\text{SEP}])_{[\text{CLS}]}$$

Score: $s_B^{(1)}(m, e) = \boldsymbol{y}_m \cdot \boldsymbol{y}_e$. Because $\boldsymbol{y}_e$ is independent of the query, all entity vectors are pre-computed and stored in a FAISS index; retrieval is maximum inner-product search in sub-linear time. The top $k$ entities are returned as $\mathcal{C}(m)$.

*Stage 2 — re-ranking.* A cross-encoder (identical in form to Architecture A) reorders $\mathcal{C}(m)$:

$$s_B^{(2)}(m, e) = \boldsymbol{y}_{m,e}\, \mathbf{W}, \qquad \text{input} = [\text{CLS}]\; \text{ctx}_l\; m\; \text{ctx}_r\; [\text{SEP}]\; d_e\; [\text{SEP}]$$

Final prediction: $\hat{e} = \arg\max_{e \in \mathcal{C}(m)} s_B^{(2)}(m, e)$.

The gap between bi-encoder and cross-encoder can be partially closed by knowledge distillation: the cross-encoder's softmax outputs at temperature $\tau$ serve as soft targets for the bi-encoder's training loss, $\mathcal{L} = \alpha\, \mathcal{L}_\text{student} + (1-\alpha)\, \mathcal{H}(\sigma(z_t; \tau),\, \sigma(z_s; \tau))$.

**Architecture C — Autoregressive generation** · source: GENRE, De Cao et al. (2021), §4.3

An entity $e$ identified by textual name $y = (y_1, \ldots, y_N)$ is scored by a BART encoder-decoder:

$$s_C(e \mid x) = p_\theta(y \mid x) = \prod_{i=1}^{N} p_\theta(y_i \mid y_{<i},\, x)$$

At each decoding step, a prefix trie $\mathcal{T}$ over all valid entity names constrains the next-token distribution: tokens not admissible as continuations of the current prefix have their log-probability set to $-\infty$. This guarantees every beam hypothesis terminates at a real entity identifier without requiring a dense entity index.

The exact softmax at each step is over vocabulary tokens (size $\sim$50K), not entity count ($\sim$6M), so no negative sampling is required during training.

| Architecture | Scoring function | Index required | Inference cost |
|---|---|---|---|
| A — Cross-encoder | $\boldsymbol{w}^\top \boldsymbol{h}_{m,e}$ | None | $O(\lvert\mathcal{C}(m)\rvert)$ forward passes |
| B — Bi-encoder + reranker | $\boldsymbol{y}_m \cdot \boldsymbol{y}_e$ then $\boldsymbol{y}_{m,e}\mathbf{W}$ | FAISS dense index | $O(1)$ retrieval + $O(k)$ reranks |
| C — Autoregressive | $\prod_i p_\theta(y_i \mid y_{<i}, x)$ | Prefix trie | $O(\text{beam} \times \bar{L})$ decoding steps |
