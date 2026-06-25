# Seeded / Crystallization Architectures

**Topic:** Architectures that begin from a small set of high-confidence seed locations — tokens, pixels, clicks, or entities — and grow or expand them into complete structured outputs (spans, masks, entity sets). Distinguished from span-enumeration or region-proposal methods that generate candidates *ex nihilo*: crystallization methods start from something *known* and propagate outward. The paradigm appears across NLP (few-shot NER, entity set expansion) and vision (weakly-supervised segmentation, interactive segmentation).

**Papers covered:** 7 papers — SEE-Few (Yang et al. 2022), Locate and Label (Shen et al. 2021), DSRG (Huang et al. 2018), SimpleClick (Liu et al. 2023), DynaMITe (Rana et al. 2023), Learning to Bootstrap (Yan et al. 2019), Global Bootstrapping Neural Network (Yan et al. 2020).

---

## 1. The Crystallization Paradigm

A crystallization architecture has three structural components:

1. **Seed identification** — find a set $\mathcal{S}$ of high-precision, low-recall anchors. Seeds are small (a few tokens, pixels, clicks, or entities) but reliable: they are known or highly likely to be part of the target structure.
2. **Expansion** — grow from $\mathcal{S}$ outward using a learned or computed similarity/coverage criterion, absorbing neighboring positions that are consistent with the seed's context.
3. **Termination** — decide when to stop expanding. This may be a learned threshold, a confidence score, a topological constraint (e.g., entity boundary), or an external signal (user correction).

This contrasts with **enumeration-then-filter** architectures (biaffine span scoring, object detection with NMS), where the full candidate space is generated first and then pruned. In crystallization, the candidate space is never enumerated: growth is local and conditioned on the seed.

The paradigm is named after seeded crystal growth in materials science: a small seed crystal in a supersaturated solution grows by accumulating atoms at its surface, guided by the lattice structure of the seed itself.

---

## 2. NLP: Few-Shot Named Entity Recognition

### SEE-Few — Yang et al., COLING 2022 · `seefew_seed_expand_entail_fewshot_named`

**Core idea.** Few-shot NER cannot rely on dense supervision to enumerate and score all spans. Instead, SEE-Few decomposes the problem into three stages trained jointly: locate a seed token sequence inside each candidate span, expand its boundaries by offset regression, then classify the expanded span via NLI entailment.

**Seeding with IoF.** The seed signal is not binary (entity / not-entity) but the **Intersection over Foreground** score:

$$\text{IoF}(A, B) = \frac{|A \cap B|}{|A|}$$

where $A$ is the candidate span and $B$ is a ground-truth entity. IoF = 1 when the candidate is fully covered by an entity (a perfect seed); IoF > 0 for any partial overlap. This gives partial credit to near-miss spans and avoids the hard negative problem: every overlapping span receives a positive gradient proportional to its coverage. The seeding loss is SmoothL1 regression to the IoF target.

**Expansion.** Seeds are expanded within a window of radius $2\lambda$ tokens:

$$o_i = \lambda \cdot (2 \cdot \text{Sigmoid}(\text{MLP}_e(h_i^{\text{exp}})) - 1) \in \mathbb{R}^2$$

predicting signed left and right boundary shifts. SmoothL1 loss against ground-truth offset $\delta$.

**Entailment classification.** The expanded span is paired with a natural-language hypothesis $\{c_i, \text{is}, \text{a}, t_j\}$ and scored as NLI — reusing the reasoning capability of pretrained language models for type discrimination.

**Joint loss:** $\mathcal{L} = \beta_1 \mathcal{L}_{\text{seed}} + \beta_2 \mathcal{L}_{\text{exp}} + \beta_3 \mathcal{L}_{\text{entail}}$

Results (K=20): CoNLL03 = 68.21, MIT-Restaurant = 60.75, WikiGold = 63.19, Weibo = 57.21 (entity-level span F1).

---

### Locate and Label — Shen et al., ACL 2021 · `locate_label_twostage_identifier_nested_named`

**Core idea.** Two-stage nested NER with explicit locate-then-label decomposition. The proposal module seeds the second stage with refined span hypotheses; the classifier is trained with IoU-weighted cross-entropy to handle imprecise proposals gracefully.

**Filter (seed selection).** Enumerate candidate spans; score each with a sigmoid filter head using max-pooling + boundary token representations. Focal loss handles class imbalance. Only spans passing threshold enter Stage 2.

**Boundary regression (expansion).** Surviving spans are boundary-regressed using a regressor that peeks one token beyond the current boundary (looking outside the seed):

$$t_i \in \mathbb{R}^2, \qquad \tilde{st} = st + \lfloor t^l + \tfrac{1}{2} \rfloor, \quad \tilde{ed} = ed + \lfloor t^r + \tfrac{1}{2} \rfloor$$

Regression loss = F1 loss + overlap loss, rewarding maximum boundary coverage.

**Classification with soft supervision.** Each proposal's cross-entropy loss is weighted by its IoU with the assigned ground-truth entity — spans that the regressor aligned well contribute more gradient, preventing the classifier from learning to type imprecise spans with full confidence.

**Soft-NMS decoding.** Rather than suppressing overlapping predictions, confidence scores are decayed proportionally to overlap with higher-scoring survivors, allowing nested entities to survive.

Results (entity-level F1): ACE04 = 87.41, ACE05 = 86.67, KBP17 = 84.05, GENIA = 80.54.

---

## 3. NLP: Entity Set Expansion *(set crystallization — inspirational, does not directly solve span finding)*

> **Note.** The two papers in this section grow a *set of known category members* across a corpus, not the *spatial extent of a single instance* within a sentence or image. The seed is a handful of known entities; the output is more entities of the same type. This is structurally analogous to spatial crystallization but solves a different problem: category membership expansion rather than boundary finding. Neither LTB nor GBN can be applied directly to span or region extraction, but both are valuable as existence proofs that seeded growth is principled and effective even with extremely sparse supervision, and their representation and growth mechanisms (distributional similarity, graph propagation, autoregressive expansion) offer design inspiration.

### Learning to Bootstrap — Yan et al., EMNLP 2019 · `learning_bootstrap_entity_set_expansion`

**Core idea.** The direct predecessor of GBN, and the paper that introduces the *delayed feedback* framing for ESE. Classic bootstrapping selects patterns by instant feedback (how good are the entities extracted right now?), but a pattern's true value is its delayed feedback — how good are the entities extracted over the next several iterations? LTB estimates delayed feedback via Monte Carlo Tree Search (MCTS) and scores entities via a Pattern Mover Similarity Network (PMSN).

**What bootstrapping is.** ESE bootstrapping is an iterative self-training loop that requires no labeled corpus — only a handful of seed entities. Running example: seeds = {London, Paris, Beijing} for the category *capital cities*. The loop alternates:
1. Scan the corpus for sentences containing known entities; extract surrounding contexts as patterns (replace the entity with `*`)
2. Score patterns by specificity to the category; keep the best ones
3. Use selected patterns to fill in `*` from the whole corpus — these are new candidate entities
4. Score candidates by similarity to known entities; add the top ones to the known set
5. Repeat with the enlarged set

From {London, Paris, Beijing} you might extract `"* is the capital of"` (specific, high-score) and `"* is a big city"` (generic, lower score). `"* is the capital of"` fills to {Moscow, Berlin, Tokyo, Ottawa} — all correct. `"* is a big city"` fills to {Shanghai, New York, Chicago, ...} — plausible but polluting.

**The delayed feedback problem.** Both patterns might look acceptable in iteration 1. But `"* is a big city"` adds Shanghai and Chicago, which generate new patterns like `"* has a population of"` and `"* is located in"` — patterns that match almost anything. Within two iterations the entity set drifts to any large city, then any city, then anything. This is *semantic drift*: a bad pattern poisons future iterations even if it looks harmless now. Instant feedback (quality of entities extracted right now) cannot distinguish `"* is the capital of"` from `"* is a big city"`; delayed feedback (quality of the entity set three iterations from now if you choose this pattern) can.

**MCTS for pattern evaluation.** Bootstrapping is cast as a tree search where each node is the current known entity set $s$ and each edge is a pattern selection $p$. Before committing to a pattern, MCTS runs many simulations, each speculatively extending the tree by several more iterations and observing the outcome. Patterns are selected during simulation by UCB:

$$p^i = \arg\max_p Q(s,p) + \mu(s,p), \qquad \mu(s,p) \propto \frac{p_\sigma(s,p)}{1 + N(s,p)}$$

$Q(s,p)$ is the accumulated reward from all previous simulations that went through edge $p$; the exploration bonus $\mu$ is large for patterns tried infrequently ($N$ small) or favoured by the prior policy $p_\sigma$. The reward at a leaf node is:

$$R = \frac{\sum_{e \in E'} \text{SIM}(e, E_0)}{|E'|} \cdot \sigma\!\left(\frac{|E'|}{a}\right)$$

The first factor is the average similarity of extracted entities $E'$ to the root seed set $E_0$ — how capital-like are the new entities? The second factor is a sigmoid on extraction count — patterns that extract only 1 entity (overly precise) or thousands (noisy) are penalised. Rewards are backed up through the tree, updating $Q(s,p)$ for every edge on the simulation path. After all simulations, the pattern with the highest $Q$ is selected for the actual bootstrapping step. In the running example: `"* is the capital of"` accumulates high reward across simulations because its downstream extractions stay capital-like; `"* is a big city"` accumulates low reward because its downstream simulations drift. MCTS distinguishes them even though their instant rewards were similar. The top-$k=200$ patterns by the classical RlogF heuristic enter MCTS to keep the branching factor tractable.

**Pattern Mover Similarity Network (PMSN).** MCTS needs a similarity function $\text{SIM}(e, E_0)$ to evaluate candidate entities. PMSN computes this without any trained neural network — it is an entirely algorithmic pipeline over corpus statistics and frozen GloVe embeddings.

Each entity is represented as a **distributional pattern embedding (DPE)**: a probability distribution over its context patterns, weighted by specificity:

$$w(p,e) = \frac{N(e,p) \times \log N(e,p)}{C(p)}$$

$N(e,p)$ is how often pattern $p$ matches entity $e$ in the corpus; $C(p)$ is how many distinct entities $p$ matches (an IDF-like denominator that downweights generic patterns). For *Moscow*: `"* is the capital of"` has small $C(p)$ (few entities fit) so high weight; `"* is a big city"` has large $C(p)$ so low weight, even if raw frequency is higher. The DPE is a matrix $X \in \mathbb{R}^{n \times d}$ of GloVe embeddings for the top-$n$ patterns paired with the normalized weight vector $w$.

Similarity between two DPEs is **Pattern Mover Similarity** — a maximum-weight bipartite matching between pattern vectors:

$$\text{PMS} = \max_{T \geq 0} \sum_{i,j} T_{ij} \cdot \text{sim}(i,j) \quad \text{s.t.} \quad \sum_j T_{ij} = w_i,\; \sum_i T_{ij} = w_j$$

$T_{ij}$ is how much weight mass is "transported" from Moscow's $i$-th pattern to the seed set's $j$-th pattern. The constraints say all mass must be shipped (proper transport). The objective finds the best possible matching: Moscow's high-weight pattern `"* is the capital of"` gets matched to the seed set's same high-weight pattern — perfect match, maximum contribution. Chicago's high-weight pattern `"* has a population of"` finds no good counterpart in the capital-city seeds — low PMS score. This is solved as a linear program at inference time; there is no gradient descent anywhere.

**Nothing in LTB is trained.** GloVe embeddings are frozen and pretrained. The importance weights $w(p,e)$ are computed analytically from corpus counts. PMS is a linear program. The only "update" during bootstrapping is a multiplicative reweighting $w_t(p,e) \propto w_{t-1}(p,e) \cdot Q(s_0,p)$ — patterns that led to good MCTS rewards get higher importance scores in subsequent iterations. This is a heuristic rule, not gradient descent. LTB is entirely an algorithmic pipeline: frozen embeddings + corpus statistics + linear programming + tree search. The word "network" in PMSN refers to the graph of entity–pattern connections, not a neural network with learnable weights. This is what makes GBN (2020) a genuine advance: it replaces the whole pipeline with a GNN and GRU trained end-to-end by backpropagation.

Results: +41% P@100, +35% P@200, +45% MAP over classical baselines on Google Web 1T; outperforms SetExpan (neural baseline) on AP/Reuters and Wiki corpora.

---

### Global Bootstrapping Neural Network — Yan et al., EMNLP Findings 2020 · `global_bootstrapping_neural_network_entity_set`

**Core idea.** ESE is the purest crystallization task in NLP: literally starting from seed entities and growing the set. GBN provides global-sighted entity representations via graph augmentation, then decodes expansions autoregressively.

**Augmented bipartite graph.** Entities and patterns form a bipartite graph. Long-tail entities have few direct links; augmented links connect entity–pattern pairs with $\geq 2$ paths of $\leq 2$ hops, giving sparse nodes access to globally informative patterns. A multi-layer GNN propagates representations through both original and augmented links:

$$h_i^l = \sigma\!\left(h_i^{l-1} + \sum_{j \in N(i)} a_j^{l-1} h_j^{l-1}\right)$$

with attention weights $a_j$ sensitive to distance (direct vs. augmented link) and link position type (before/middle/after).

**Autoregressive expansion (GBDecoder).** A GRU maintains the category embedding $h_c^t$, updated at each step by an attention-weighted summary of the last expansion batch. New entities are ranked by cosine similarity to $h_c^t$ and the top-$N$ are added. The loop continues until a termination condition is met.

**Self-supervised pretraining.** Because seeds are sparse, GBEncoder is pretrained with neighborhood contrastive learning (nearby nodes should be similar) and masked link prediction before fine-tuning on the actual ESE task.

Results: state-of-the-art precision–throughput and P@Iter.K on CoNLL 2003 and OntoNotes ESE benchmarks.

---

## 4. Vision: Weakly-Supervised Segmentation

### DSRG — Huang et al., CVPR 2018 · `weaklysupervised_semantic_segmentation_network_deep_seeded`

**Core idea.** The vision archetype of crystallization: CAM-derived discriminative pixels are seeds; they grow into complete object masks via a region-growing algorithm that uses the segmentation network's own output as the similarity criterion, creating a self-improving loop.

**Seed generation.** A classification network (VGG-16 or ResNet-101) generates Class Activation Maps. Top-20% pixels per class heatmap = foreground seeds $S_c$. Pixels below normalized saliency 0.06 = background seeds.

**Seeding loss.** The segmentation network is trained only at seeded positions (non-seeded pixels contribute zero gradient):

$$\ell_{\text{seed}} = -\frac{1}{\sum_c |S_c|} \sum_{c \in \mathcal{C}} \sum_{u \in S_c} \log H_{u,c}$$

**Deep Seeded Region Growing.** After each forward pass, the segmentation probability map $H$ is used as the growing criterion. Pixel $u'$ joins class $c$ if $H_{u',c} \geq \theta_c$ (0.85 for foreground, 0.99 for background) and $c = \arg\max_{c'} H_{u',c'}$. The grown region $S$ becomes the new supervision for the next epoch. Original seeds are always retained to prevent drift.

**Joint loss:** $\ell = \ell_{\text{seed}} + \ell_{\text{boundary}}$

**The self-improving loop in detail.** $H$ is the network's full probability map over all pixels and all classes — a score $H_{u,c}$ for every (pixel, class) pair. The seeding loss is standard cross-entropy, but evaluated only at seeded pixels $S_c$; non-seeded pixels contribute zero gradient. The network is not told it is wrong anywhere else — it simply receives no signal there. After each epoch, DSRG runs a growing pass: for each unseeded pixel $u'$, add it to class $c$ if the network is confident ($H_{u',c} \geq \theta_c$) and $c$ is the top prediction. The grown region becomes next epoch's supervision. The threshold is asymmetric: $\theta_{\text{foreground}} = 0.85$, $\theta_{\text{background}} = 0.99$ — background is kept strict because it covers most of the image and a loose threshold would absorb foreground pixels irreversibly. Original CAM seeds are permanently retained in $S$ to prevent drift: they act as an anchor that forces the network to stay correct at the most reliable pixels even as the grown region expands. The loop converges because growing only happens where the network is already confident, and training on those grown pixels makes the network more confident in adjacent regions — a self-reinforcing expansion from seeds outward. The boundary loss ($\ell_{\text{boundary}}$, a CRF term) sharpens the grown masks to follow actual image edges rather than bleeding across object boundaries.

Results: PASCAL VOC 2012 test — VGG16: 60.4 mIoU; ResNet101: 63.2 mIoU (SOTA at publication among image-label-only methods).

This is self-reinforcing crystallization: the crystal (segmentation network) generates the growth medium (pixel-level pseudo-labels), which feeds the next crystal growth step.

---

## 5. Vision: Interactive Segmentation

### SimpleClick — Liu et al., ICCV 2023 · `simpleclick_interactive_image_segmentation_simple_vision`

**Core idea.** User clicks are seeds; a plain MAE-pretrained ViT grows a segmentation mask around them via global self-attention, without any hierarchical backbone or explicit region-growing algorithm.

**Click encoding.** Clicks are encoded in a two-channel disk map (positive/negative), fused with the previous mask, and injected into the backbone via a symmetric patch embedding layer added to the image patch embedding. The ViT's full self-attention then propagates click information globally — every patch "sees" every click location.

**Feature pyramid from last layer only.** Because all ViT feature maps are at the same resolution, SimpleClick applies four parallel conv/deconv layers to the *last* feature map to produce a simple multi-scale pyramid at $\{1/32, 1/16, 1/8, 1/4\}$ scale. An MLP segmentation head fuses the pyramid into a binary mask prediction.

**Iterative refinement.** Training simulates clicks iteratively: the next click is placed at the largest error region of the current prediction. This creates a curriculum where each click corrects a specific failure, matching the crystallization metaphor — subsequent seeds patch holes in the existing crystal.

Results: ViT-H achieves 4.15 NoC@90 on SBD (−21.8% clicks vs. prior best). Strong out-of-domain generalization to medical imaging.

---

### DynaMITe — Rana et al., ICCV 2023 · `dynamite_dynamic_query_bootstrapping_multiobject_interactive`

**Core idea.** Extends interactive segmentation to *multiple objects jointly*. Each click seeds an instance-specific Transformer query bootstrapped from the backbone features at the click location; queries are updated at each refinement step via masked cross-attention without re-computing image features.

**Query bootstrapping (seed → query).** For click $c_j$, the query is:

$$q_j = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} f_{c_j}$$

i.e., the average of backbone features at the click location across all scales. The query is grounded in visual content at the seed from initialization.

**Masked cross-attention (controlled growth).** The encoder restricts each query's cross-attention to pixels currently within its mask prediction $\mathcal{M}^{l-1}$:

$$Q_l \leftarrow \text{softmax}(\mathcal{M}_{l-1} \odot Q_l K_l^T) V_l + Q_{l-1}$$

This is growth with a boundary: the query expands only into the region it already claims, preventing cross-contamination between nearby objects.

**Multi-step refinement without re-encoding.** When a new click $c_{t+1}$ arrives, its query is bootstrapped and appended to $Q^t$. The backbone is not re-run. Only the Transformer processes the new query, making multi-step correction $O(\text{Transformer})$ rather than $O(\text{backbone})$ per click.

Results: SOTA on standard single-instance benchmarks; outperforms sequential single-instance methods on the proposed multi-instance MIST benchmark using fewer total clicks per image.

---

## 6. Connecting Threads

### The shared structure

| Component | SEE-Few | Locate & Label | LTB | GBN | DSRG | SimpleClick | DynaMITe |
|---|---|---|---|---|---|---|---|
| **Seed signal** | IoF score from BERT span | Filtered proposals (filter head) | Seed entity set | Seed entity set | CAM activations | User click disk map | User click location features |
| **Growth criterion** | MLP expansion head | Boundary offset regressor | MCTS + PMS similarity | GRU + cosine similarity | Segmentation prob. map | Global ViT self-attention | Masked cross-attention |
| **Supervision of growth** | SmoothL1 vs. offset | F1 + overlap loss | Delayed feedback reward $R$ | Category embedding update | Seeding loss (at seeds only) | NFL click simulation | NFL click simulation |
| **Termination** | Fixed window $2\lambda$ | Threshold | Fixed MCTS budget | N entities per step | Probability threshold $\theta$ | Fixed click budget | Fixed iteration budget |
| **Output** | Entity span + type | Entity span + type | Expanded entity set | Expanded entity set | Pixel-level class mask | Binary instance mask | Multi-object masks |

### Two kinds of crystallization: spatial vs. set

The papers in this file fall into two structurally distinct variants of the crystallization paradigm:

**Spatial crystallization** (SEE-Few, Locate & Label, DSRG, SimpleClick, DynaMITe) — the seed is a location or region *inside a single input* (a sentence, an image). Growth expands the seed's boundaries outward within that input until it covers the full extent of one entity or object. The output is a span or mask — a contiguous region. This directly solves span/region finding.

**Set crystallization** (LTB, GBN) — the seed is a handful of known members of a category. Growth finds more members of the same category across a corpus. There are no spatial boundaries to find; the entity strings are already complete. The output is an expanded set of category instances. This does **not** directly solve span or region finding.

The distinction matters for our work: spatial crystallization methods are directly applicable to the problem of finding entity spans from seed tokens. Set crystallization methods are inspirational — they demonstrate that seeded growth under sparse supervision is principled and scales well — but their machinery (corpus-wide graph propagation, set membership scoring) would need fundamental redesign to address spatial boundary finding. The most transferable idea from LTB/GBN is the delayed feedback intuition: the quality of a seed's expansion should be judged by the downstream quality of what gets found, not just by immediate coverage.

### Why not enumerate?

Span-enumeration NER (biaffine scoring, BOPN, NER-as-OD) and full-image segmentation both enumerate the full output space before filtering. This is tractable when supervision is dense and the model can learn to confidently score all positions. Crystallization is advantageous when:
- **Supervision is sparse**: few-shot NER (SEE-Few), image-level labels only (DSRG), few seed entities (GBN). Growing from high-confidence seeds avoids committing to the full space before the model is ready.
- **Candidate space is too large**: $N^2$ spans (Locate & Label), $H \times W$ pixels (interactive segmentation). Starting from seeds prunes the search to regions near known anchors.
- **Outputs are structured and contiguous**: entities and object masks are spatially/sequentially connected. Crystallization exploits this: growth is local and bounded, which is a strong inductive bias.

### IoF vs. IoU in seed supervision

SEE-Few's choice of IoF over IoU for seed scoring is instructive. IoU penalises large seeds that extend beyond the entity boundary; IoF rewards seeds that are fully contained, regardless of size. For *seeding* (finding tokens inside an entity), containment is the right criterion — you want seeds to be inside the target, not to match it exactly. This is the NLP analogue of DSRG's seeding loss ignoring non-seeded pixels rather than penalising them.

### Self-improving loops

DSRG and GBN both exhibit self-improving crystallization: the model's own output at step $t$ becomes the supervision/growth signal for step $t+1$. This requires careful stabilisation (DSRG always retains original seeds; GBN uses a teacher model to filter expansions). SimpleClick and DynaMITe achieve a similar effect through iterative click simulation during training, where the model learns to grow from its own imperfect masks.

### Cross-domain applicability

The crystallization paradigm generalises across modalities because the abstract structure (seed → grow → refine) is independent of the representation. Seeds are tokens in NLP and pixels/clicks in vision; the similarity/expansion criterion is a learned MLP, a graph attention, or a Transformer cross-attention; the output is a span, a mask, or an entity set. Any task where high-confidence anchors are available and outputs are spatially/semantically contiguous around those anchors is a candidate for crystallization.
