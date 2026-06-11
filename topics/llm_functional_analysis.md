# LLMs as Mathematical Functions: Invariants, Geometry, and Invertibility

*Papers in this cluster treat language models not as engineering artefacts but as mathematical objects — asking what structural properties the map from inputs to representations necessarily has, and what geometric invariants are preserved or destroyed across that map. A secondary thread concerns how to measure and compare representation spaces across models, and what convergence of those spaces implies.*

---

## The central question

The papers in this cluster share a single orienting question: **what kind of function is a trained language model?** Rather than asking what a model can do on a benchmark, they ask what can be formally proved or empirically reverse-engineered about the map $f: \text{input} \to \text{representation}$ (or $f: \text{input} \to \text{output}$) as a mathematical object. The questions range from topological (is $f$ injective? invertible?) to algebraic (what algorithm does $f$ implement?) to geometric (what invariants does the representation space carry?). A secondary set of questions asks: given that we want to compare or align two representation spaces, what is the right notion of similarity — and what does empirical convergence of those spaces imply about what models are learning?

---

## How the papers relate

**`language_models_injective_hence_invertible`** (Nikolaou et al., ICLR 2026) is the most purely mathematical: it proves that decoder-only transformers are almost-surely *injective* — different input sequences produce different last-token hidden states — and from this derives *invertibility*: the algorithm SIPIT recovers the exact input from hidden activations in linear time. This is a structural theorem about the function class, independent of training.

**`progress_measures_grokking_mechanistic_interpretability`** (Nanda et al., ICLR 2023) takes the opposite approach: not top-down proof but bottom-up reverse engineering. For a one-layer transformer trained on modular addition $a + b \bmod P$, the authors fully reconstruct the algorithm implemented in the weights. The model maps inputs to a sparse Fourier basis $(\sin(w_k a), \cos(w_k a))$, computes addition as rotation on the circle, and reads off the result by inner product. The invariant here is Fourier-algebraic: the key frequencies $w_k = \frac{2\pi k}{P}$ are preserved across the full forward pass. Grokking, viewed through this lens, is the training dynamics gradually amplifying this invariant structure while eliminating competing memorisation components.

**`emergent_world_representations_exploring_sequence_model`** (Li et al., ICLR 2023) asks whether invariant structure emerges from sequence modeling even when not explicitly required. Training a GPT variant on Othello moves with no knowledge of rules, they find that the hidden states encode a nonlinear representation of the board state — a latent variable that is an invariant of the game-play process, not of the token sequence surface form. Crucially, this representation is *causally active*: editing it changes model outputs in the predicted direction. The board state is thus something the model's internal function preserves across its computation.

**`toy_models_superposition`** (Elhage et al., Transformer Circuits Thread 2022) asks a prior question: given that features are represented as directions, *how many* can be stored? In toy ReLU models with $n$ features compressed into $m \ll n$ dimensions, the answer depends on feature sparsity. When features are dense, the model behaves like PCA and represents only the top $m$ orthogonally. When features are sparse, it pays to tolerate small mutual interference and pack additional features as almost-orthogonal directions — superposition. The geometry of the resulting configurations mirrors the Thomson problem: features arrange themselves as vertices of uniform polytopes (antipodal pairs, triangles, pentagons, tetrahedra). The transition between regimes is a sharp first-order phase change in the sparsity–importance plane, and the formal criterion is that $W^T W$ is non-invertible. This explains polysemanticity as an inevitable consequence of capacity pressure, and connects directly to both the grokking dynamics (superposition dissolving during cleanup) and the linear representation geometry (the inner product structure matters most when features are not in superposition).

**`linear_representation_hypothesis_geometry_large_language`** (Park et al., ICML 2024) operates at the level of the representation space geometry rather than any individual algorithm. It formalises what it means for a concept to be *linearly* encoded — by two definitions: in the unembedding space (connected to probing/measurement) and in the embedding space (connected to steering/intervention) — and proves these are unified by a single *causal inner product* estimable from the unembedding matrix. The paper thus characterises what geometric structure the representation space must have if the linear representation hypothesis holds: concepts are orthogonal directions under this inner product, not the standard Euclidean one.

**`language_models_represent_space_time`** (Gurnee & Tegmark, ICLR 2024) provides a large-scale empirical test of the linear representation hypothesis for a specific class of features: real-valued spatial (latitude/longitude) and temporal (date) coordinates. Using ridge regression probes on Llama-2 (7B–70B) activations over six datasets totalling ~187K named entities, they show that spatial and temporal features are linearly decodable, plateau at mid-depth, generalise across entity types, are robust to prompting, and are causally active — individual "space neurons" and "time neurons" can be found whose targeted editing shifts model outputs in the predicted direction. The paper is notable for treating *continuous, metric-valued* world features rather than categorical concepts, and for the causal verification step that distinguishes genuine representation from epiphenomenal correlation.

**`similarity_neural_network_representations_revisited`** (Kornblith et al., ICML 2019) addresses the methodological question underlying all cross-model representational comparison: what is the right similarity measure for neural network activation spaces? The key theoretical result is a no-go theorem: any index invariant to *invertible linear transformation* gives the same score to any two representations of width ≥ dataset size, collapsing to a useless constant. The correct invariance properties are orthogonal transformation and isotropic scaling — not full linear equivalence. This motivates *Centered Kernel Alignment* (CKA):

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

which compares the *similarity structures* (kernels, or representational similarity matrices) induced by two representations rather than their coordinate vectors directly. CKA is the formal realisation of the intuition that what matters is the relational structure among datapoints — not the absolute coordinates in the representation space.

**`platonic_representation_hypothesis`** (Huh et al., ICML 2024) takes the framework developed in Kornblith et al. and uses it to ask a much larger question: are all neural networks converging to the *same* relational structure? Using mutual nearest-neighbor metrics over kernels, the paper demonstrates that larger and more capable models (both language and vision) are increasingly aligned with each other, and that language and vision representations align cross-modally in proportion to model capability. The hypothesis is that this convergence has an endpoint — the platonic representation, $K^*$ — which reflects the joint distribution of events in the world that generated the training data. Every model is a projection of this underlying reality, and training pushes the kernel toward $K^*$. This directly formalises the intuition that models may be learning a relational graph rather than an absolute coordinate system: the kernel $K(x_i, x_j)$ is exactly the coordinate-free relational content, and it is this relational content that is converging.

Together, the papers form an extended logical arc:

1. The function $f$ is injective and invertible (Nikolaou et al.) — structure at the level of the full map.
2. For specific tasks, the algorithm $f$ implements can be fully reconstructed from weights (Nanda et al.) — structure at the level of the computation.
3. Even without explicit supervision, $f$ preserves latent state variables of the generating process (Li et al.) — structure at the level of emergent invariants.
4. The capacity of the representation to store features is governed by sparsity, via a phase transition (Elhage et al.) — structure at the level of the hidden-state geometry.
5. The range of $f$ has a non-Euclidean geometry that respects causal structure (Park et al.) — structure at the level of the output space.
6. Specific metric world features (space, time) are linearly encoded and causally active (Gurnee & Tegmark) — concrete empirical instances of the linear representation hypothesis.
7. The right way to compare two representation spaces is via their induced kernels, not their coordinates; the correct invariance is orthogonal + isotropic, not full linear (Kornblith et al.) — mathematical foundation for all cross-model comparison.
8. Representation spaces across architectures, objectives, and modalities are converging toward a shared kernel reflecting world structure (Huh et al.) — a convergence hypothesis that frames the relational kernel as the true object of representation learning.

---

## The metric question: which distance for BERT-class models?

The question of what distance metric is appropriate for RDMs computed from transformer activations connects several papers in this cluster and remains partly open.

`similarity_neural_network_representations_revisited` establishes that any metric invariant to the full invertible linear group is degenerate when model width exceeds dataset size — ruling out CCA-based approaches and motivating CKA. But CKA still uses an isotropic kernel; it strips out all coordinate information including the information encoded by the model's own weight matrices.

`curved_spacetime_transformer_architectures` makes the strongest case that the *correct* metric is not isotropic at all: the Q/K weight matrices of each attention head define an effective bilinear metric $g_{ij} = x_i^T (W^Q)^T W^K x_j$, which is the actual relational structure the model computes over. Standard cosine similarity implicitly replaces this with the identity matrix. The paper does not translate this into a practical RSA methodology but establishes the theoretical target.

`visualizing_measuring_geometry_bert` shows (via the mathematical argument for why squared $L_2$ distance recovers parse tree distances) that different geometric questions call for different metrics even within BERT: tree-structural distance requires a learned linear-projection metric (Hewitt & Manning's structural probe), while semantic subspace analysis uses PCA projections. Neither is simply cosine distance.

`linear_representation_hypothesis_geometry_large_language` resolves one piece: the causal inner product (weighted by the unembedding matrix) is the correct metric for measuring concept direction geometry at the output layer. This is architecture-derived but specific to the readout side.

What has not been done: a systematic derivation of the layer-appropriate metric for *intermediate* BERT layers using the downstream Q/K/V/MLP weight matrices from layers $\ell+1$ onward — the metric that captures what the model's remaining computation will do with a representation at layer $\ell$.

**The Structural Probe (Hewitt & Manning, NAACL 2019)** is the closest existing work: it learns a weight matrix $B$ such that $\|(u-v)B\|^2$ recovers parse tree distance, showing the right metric is task-derived and not isotropic. The paper has no arXiv preprint; PDF available from ACL Anthology (N19-1419) — obtain manually and add to the bank.

## Open threads

- The Fourier circuit in `progress_measures_grokking_mechanistic_interpretability` is exact for modular arithmetic; whether analogous algebraic invariants exist for natural-language tasks is open.
- `emergent_world_representations_exploring_sequence_model` finds a *nonlinear* world representation, whereas `linear_representation_hypothesis_geometry_large_language` and `language_models_represent_space_time` both find *linear* concept/feature representations — reconciling these geometries (and when each applies) is unresolved.
- The injectivity result in `language_models_injective_hence_invertible` holds almost-surely; characterising the measure-zero set of collisions, and its implications for representational geometry, connects to the causal inner product construction.
- `toy_models_superposition` establishes the phase-change picture for toy ReLU networks; how the superposition geometry interacts with the causal inner product of `linear_representation_hypothesis_geometry_large_language` in real LLMs is an open question.
- `platonic_representation_hypothesis` proves convergence of relational structure across models, but says little about what the specific geometric features of $K^*$ are; connecting this to the linear representation hypothesis and the causal inner product would require identifying which features are universal across the converged kernel.
- `similarity_neural_network_representations_revisited` shows early layers align across datasets while late layers diverge; `language_models_represent_space_time` finds spatial/temporal representations plateau at mid-depth — characterising what different depth regimes encode structurally is open.
- `geometry_hidden_representations_large_transformer_models` identifies the expand–compress–decode ID profile as a layer-selection heuristic; how this profile interacts with the stability pattern in the user's binding-strength experiments (does stability peak where ID is compressed?) is untested.
- **Architecture-appropriate RDM metric for BERT** (open problem): deriving the layer-$\ell$ metric from downstream weight matrices $W^Q_{\ell'}, W^K_{\ell'}, W^V_{\ell'}, W^O_{\ell'}$ for $\ell' > \ell$ has not been done. `curved_spacetime_transformer_architectures` provides the theoretical vocabulary; the Structural Probe provides the supervised precedent; filling this gap would put experiments like the binding-strength stability work on an architecturally grounded footing.
- "A Mathematical Framework for Transformer Circuits" (Elhage et al. 2021) — not on arXiv (transformer-circuits.pub). Obtain manually.
- "A Structural Probe for Finding Syntax in Word Representations" (Hewitt & Manning, NAACL 2019) — no arXiv preprint. Obtain from ACL Anthology N19-1419.

---

## Key papers in this cluster

| id | year | one-liner |
|----|------|-----------|
| `language_models_injective_hence_invertible` | 2026 | Proves decoder-only LMs are almost-surely injective and introduces SIPIT for exact input recovery from hidden states. |
| `progress_measures_grokking_mechanistic_interpretability` | 2023 | Reverse-engineers the Fourier-basis rotation circuit for modular addition and decomposes grokking into three continuous training phases. |
| `emergent_world_representations_exploring_sequence_model` | 2023 | A GPT trained on Othello transcripts develops a causal nonlinear representation of the board state without explicit supervision. |
| `toy_models_superposition` | 2022 | Demonstrates that sparse features are stored as almost-orthogonal directions beyond model dimensionality, with a first-order phase transition governing when superposition occurs. |
| `linear_representation_hypothesis_geometry_large_language` | 2024 | Formalises linear concept representations via counterfactuals and introduces a causal inner product that unifies probing and steering geometries. |
| `language_models_represent_space_time` | 2024 | LLMs linearly encode metric world features (space, time) across Llama-2 scales; individual space/time neurons are causally active. |
| `similarity_neural_network_representations_revisited` | 2019 | Proves full-linear-invariance indices are degenerate; introduces CKA, which correctly captures representational similarity as kernel alignment. |
| `platonic_representation_hypothesis` | 2024 | Provides evidence that representation kernels across models, scales, and modalities are converging toward a universal relational structure reflecting world statistics. |
| `visualizing_measuring_geometry_bert` | 2019 | Syntactic structure is encoded in both BERT attention matrices and residual stream embeddings; word senses occupy a low-dimensional semantic subspace; provides mathematical argument for why squared $L_2$ distance is the natural tree-embedding metric. |
| `geometry_hidden_representations_large_transformer_models` | 2023 | Intrinsic dimension follows a universal expand–compress–decode profile across transformer layers; the compression minimum identifies semantically optimal layers without supervision. |
| `curved_spacetime_transformer_architectures` | 2025 | Q/K weight matrices define the transformer's effective bilinear metric; attention implements parallel transport; empirical curvature tests confirm token trajectories follow curved paths through representation space. |
