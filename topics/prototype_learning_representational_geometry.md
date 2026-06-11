# Prototype Learning and Representational Geometry in NLP

*Papers in this cluster examine how representations from pretrained language models should be aggregated into prototypes, what distance function is appropriate for comparing them, and how the known geometric pathologies of transformer representations interact with prototype-based classifiers. The running example is few-shot named entity recognition (NER), where the target is formally defined — classify a span as an entity type by its distance to a class prototype — which makes the metric question answerable from first principles rather than by convention.*

---

## The central question

Prototype-based classification is seductively simple: represent each class by a single vector (the prototype), then assign a query to the nearest class. Two design choices are entailed — how to construct the prototype from support examples, and how to measure distance at test time — and they are not independent. The literature mostly treats both as defaults (mean aggregation; cosine or Euclidean distance), without asking whether these defaults are internally consistent or statistically optimal. This cluster traces the derivation from first principles and confronts it with empirical findings about the geometry of transformer representations.

---

## Step 1: When is the mean the right prototype?

`prototypical_networks_fewshot_learning` (Snell et al., NeurIPS 2017) provides the foundational formal result. A Bregman divergence derived from a strictly convex function $\psi$ is a distance of the form

$$D_\psi(z, \mu) = \psi(z) - \psi(\mu) - \langle\nabla\psi(\mu),\, z - \mu\rangle.$$

The key theorem: for any Bregman divergence, the point $\mu$ minimising the average divergence to a set of points is their arithmetic mean. Formally, $\arg\min_\mu \mathbb{E}[D_\psi(z, \mu)] = \mathbb{E}[z]$. The proof differentiates the expectation with respect to $\mu$ and uses strict convexity of $\psi$ to show the unique zero is $\mu = \mathbb{E}[z]$.

Taking $\psi(z) = \|z\|^2$ recovers squared Euclidean distance and the ordinary mean. The deeper interpretation connects to exponential families: prototypical classification with a Bregman divergence is MAP classification in a mixture model where each class generates from an exponential family distribution with mean $\mu_k$. The specific choice $\psi(z) = \|z\|^2$ corresponds to isotropic Gaussians $\mathcal{N}(\mu_k, \sigma^2 I)$ — the whole pipeline (mean prototype, squared Euclidean distance) is internally consistent under this single assumption.

**The cosine inconsistency.** Cosine distance is *not* a Bregman divergence: the point minimising average cosine distance to a set of vectors is not their arithmetic mean but rather the normalised mean (the direction of the centroid). Using arithmetic mean prototypes with cosine distance is therefore internally inconsistent — the prototype is not the Fréchet mean of the distance being used. Snell et al. verify this empirically: Euclidean distance substantially outperforms cosine in few-shot image classification.

---

## Step 2: What does the Gaussian assumption require?

The isotropic Gaussian model ($h \mid c \sim \mathcal{N}(\mu_c, \sigma^2 I)$) has two testable predictions: representations should be (1) clustered around class means, and (2) isotropically distributed around those means. The second prediction is what isotropy means: equal variance in all directions. If the within-class covariance is $\Sigma \neq \sigma^2 I$, the Bayes-optimal classifier is LDA with metric $G = \Sigma^{-1}$, not Euclidean. The mean prototype remains optimal (it is the Fréchet mean of the Mahalanobis distance for any $G$, since $G$ cancels in the gradient), but the distance changes.

`whitening_sentence_representations_better_semantics_faster` (Su et al., 2021) provides a direct empirical test: applying the full whitening transformation $h \mapsto (h - \mu)\Sigma^{-1/2}$ — which is exactly the change of metric from Euclidean to $\Sigma^{-1}$-Mahalanobis — improves semantic textual similarity from ~25 Spearman to ~74 Spearman on BERT without any fine-tuning. The improvement comes entirely from correcting the anisotropy. This validates $G = \Sigma^{-1}$ empirically as the appropriate metric for BERT-based representations in downstream distance computations.

---

## Step 3: How badly does BERT violate isotropy?

`how_contextual_contextualized_word_representations_comparing` (Ethayarajh, EMNLP 2019) quantifies the violation. Two measurements:

*Self-similarity*: for word type $w$, $\text{SS}^\ell(w) = \mathbb{E}_{c \neq c'}[\cos(v_w^{(c,\ell)}, v_w^{(c',\ell)})]$. This monotonically decreases with layer depth — upper BERT layers produce representations where the same word in different contexts has low cosine similarity. Less than 5% of the variance in a word's contextualised representations across contexts can be explained by a static (context-independent) embedding.

*Isotropy*: $I^\ell = \mathbb{E}_{v,v' \sim \text{Uniform}}[\cos(v^{(\ell)}, v'^{(\ell)})]$, which should be ~0 for isotropic distributions but is consistently 0.3–0.6 in BERT upper layers. Representations cluster in a narrow cone — all pairs have elevated cosine similarity regardless of semantic relationship.

These two measurements together explain why mean prototypes in upper BERT layers are problematic: (1) the context signal (contextual noise $\epsilon_i$) explains >95% of the variance in each representation, so the signal-to-noise ratio for extracting a class-specific prototype is very low; (2) the cone structure means cosine similarity between the prototype and query tokens is largely dominated by the global cone direction rather than class-specific information.

---

## Step 4: The rogue-dimension mechanism

`allbutthetop_simple_effective_postprocessing_word_representations` (Mu & Viswanath, ICLR 2018) identifies the geometric mechanism in static word embeddings: the top-$D$ PCA directions of the vocabulary have variance one to two orders of magnitude larger than the rest, and these directions correlate with word frequency rather than semantics. Subtracting the mean and projecting out these $D$ directions consistently improves word similarity, analogy, and sentence similarity tasks.

`all_bark_no_bite_rogue_dimensions` (Timkey & van Schijndel, EMNLP 2021) shows the same phenomenon in contextualised models: 1–3 dimensions in BERT and GPT-2 have variance hundreds of times larger than the median dimension. These "rogue dimensions" dominate cosine and Euclidean similarity — the distance between any two tokens becomes almost entirely a function of their rogue-dimension coordinates, regardless of what the other ~765 dimensions encode. Crucially, ablation shows these dimensions are not important to model behaviour: zeroing them out does not degrade task performance. The dimensions dominating the metric are orthogonal to the dimensions driving the computation. Simple per-dimension standardisation (the diagonal approximation to $\Sigma^{-1}$) recovers the underlying representational quality.

**The unified picture.** Both papers identify the same root cause — an anisotropic variance spectrum — and the same prescription — down-weight or remove the dominant directions. The difference is mechanistic: in static embeddings, the dominant directions encode frequency artefacts from co-occurrence statistics; in contextualised models, the dominant dimensions appear to arise from positional or structural signals baked in during pretraining. The fix in both cases is a special case of the Mahalanobis metric: Mu & Viswanath remove $D$ directions (setting their $\Sigma^{-1}$ eigenvalues to zero), Timkey standardises (diagonal $\Sigma^{-1}$), Su et al. apply the full $\Sigma^{-1}$.

---

## Step 5: Implications for the prototype construction problem

The previous steps establish that:
1. Mean prototype is correct if the distance is a Bregman divergence (Snell).
2. The Bayes-optimal Bregman divergence under the Gaussian model is squared Mahalanobis with $G = \Sigma^{-1}$ (LDA).
3. BERT representations have anisotropic covariance $\Sigma \neq \sigma^2 I$ (Ethayarajh, Timkey).
4. Using $G = \Sigma^{-1}$ substantially improves downstream tasks (Su et al.).

But a further problem complicates the prototype extraction: the 5%-variance result from Ethayarajh means that even in the direction of the class mean $\mu_c$, the contextual noise $\epsilon_i$ swamps the signal. The arithmetic mean $\bar{h} = \mu_c + \frac{1}{n}\sum_i \epsilon_i$ converges to $\mu_c$ only if $\mathbb{E}[\epsilon_i] = 0$ and $n$ is large enough to average out the noise. In few-shot NER, $n \sim 5$–$10$ — far too small given that within-class variance is ~20× the signal.

This motivates the NER-specific approaches:

`fewshot_named_entity_recognition_entitylevel_prototypical` (Su et al., COLING 2022) adds a *dispersion loss* that explicitly pushes prototypes of different classes apart. This addresses the case where mean prototypes of different classes collapse to nearby regions — the signal is too weak and the noise too large for the means alone to be well-separated. The dispersion loss compensates by optimising the inter-prototype geometry directly.

`spanproto_twostage_spanbased_prototypical_network_fewshot` (Wang et al., EMNLP 2022) addresses a different source of contamination: using token-level representations to build span-level prototypes introduces alignment noise (which token of a multi-token entity best represents the entity?). By operating at the span level directly — pooling representations over the span and building span-level prototypes — the contamination from non-entity tokens is reduced.

`container_fewshot_named_entity_recognition_contrastive` (Das et al., ACL 2022) takes the most radical approach: abandon class-specific prototypes entirely and instead train a category-agnostic contrastive objective that directly optimises the distance between same-type and different-type tokens. By modelling each token as a Gaussian distribution and using KL divergence as the distance, CONTaiNER simultaneously addresses the metric question ($d_\text{KL}$ replaces cosine) and the prototype contamination question (no explicit prototype is needed — the distance function is learnt end-to-end).

---

## Open threads

- **The zero-mean assumption.** The arithmetic mean converges to the class signal only if $\mathbb{E}[\epsilon_i] = 0$ across the support set. For BERT, the contextual noise $\epsilon_i$ is likely not zero-mean: entities appear in systematic syntactic positions (subjects, noun phrases) that are consistent within a class and therefore produce a non-zero mean in the noise. Empirically testing whether within-class context distributions are balanced, and by how much, is unresolved.
- **Few-shot signal-to-noise.** The 5% variance figure from Ethayarajh gives an upper bound on how much of $h_i$ the static signal explains. For a few-shot support set of $n = 5$ examples, the prototype estimate $\bar{h}$ has noise of order $\sqrt{\text{Var}(\epsilon_i) / n} \approx \sqrt{19 \cdot \text{Var}(s_c) / 5} \approx 2\|\Sigma_s^{1/2}\|$ — the noise is roughly twice the signal amplitude. How much this harms NER performance, and whether $\Sigma^{-1}$ whitening compensates fully or only partially, has not been characterised.
- **Whitening in the few-shot regime.** Su et al.'s whitening is estimated on a large corpus. In few-shot NER, a corpus-level $\hat{\Sigma}$ is available (from the pretrained model applied to unlabelled text), but whether this $\hat{\Sigma}$ is the right covariance to use (vs. the within-class $\Sigma_c$ estimated from few examples) is unclear.
- **Span-level vs. token-level geometry.** SpanProto and EP-Net build span representations by pooling token vectors. Whether the rogue-dimension structure persists at the span level (after mean pooling over 2–5 tokens) or is partially averaged out is unstudied.
- **The dispersion loss and prototype collapse.** EP-Net's dispersion loss pushes prototypes apart, which helps when the signal is weak. But it introduces a new hyperparameter (the margin) and does not address the underlying cause (weak signal relative to noise). Whether dispersion loss + better metric (whitened Mahalanobis) outperforms either alone is an open empirical question.
- **CONTaiNER as metric learning.** CONTaiNER learns the distance function end-to-end via the Gaussian embedding and KL divergence objective. This sidesteps the need to derive the right metric analytically — but it requires a training phase with entity-labelled data. In zero-label settings (pure prototype learning from a frozen encoder), the analytical derivation matters again.

---

## Key papers in this cluster

| id | year | one-liner |
|----|------|-----------|
| `prototypical_networks_fewshot_learning` | 2017 | Proves mean is the optimal prototype under any Bregman divergence; connects prototype learning to MAP inference in exponential family mixtures. |
| `how_contextual_contextualized_word_representations_comparing` | 2019 | Measures BERT's anisotropy and contextuality: <5% of variance in upper-layer representations is explained by a static embedding; cone structure inflates all cosine similarities. |
| `allbutthetop_simple_effective_postprocessing_word_representations` | 2018 | Removing the mean and top-D PCA directions from word vectors eliminates frequency artefacts and consistently improves intrinsic and extrinsic tasks. |
| `all_bark_no_bite_rogue_dimensions` | 2021 | 1–3 rogue dimensions dominate similarity measures in BERT/GPT-2 but are unimportant to model behaviour; standardisation corrects this and reveals genuine representational quality. |
| `whitening_sentence_representations_better_semantics_faster` | 2021 | Whitening BERT sentence embeddings (applying $\Sigma^{-1/2}$) is equivalent to using the Mahalanobis metric and lifts STS performance from ~25 to ~74 Spearman without fine-tuning. |
| `fewshot_named_entity_recognition_entitylevel_prototypical` | 2022 | Addresses prototype collapse in few-shot NER via a dispersion loss that explicitly maximises inter-prototype distance, with span-level encoding. |
| `spanproto_twostage_spanbased_prototypical_network_fewshot` | 2022 | Two-stage span-level prototype network for few-shot NER; reduces alignment noise by building prototypes directly from span pooling. |
| `container_fewshot_named_entity_recognition_contrastive` | 2022 | Learns a category-agnostic contrastive objective over Gaussian token embeddings, replacing both the explicit prototype and the fixed distance function with an end-to-end trained KL-divergence metric. |
