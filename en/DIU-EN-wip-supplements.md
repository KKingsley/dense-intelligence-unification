# DIU WIP — Supplementary Chapters

> Status: ✅ WIP-1~8 all complete  
> WIP-1 $\mathcal{D}_1$ internal ordering · WIP-2 compute/software stack · WIP-3 mapping theory · WIP-4 complete proofs of hierarchy transitions · WIP-5 coverage-regularized training objective · WIP-6 multimodal extension · WIP-7 relation to Shannon entropy · WIP-8 dynamic DIU training evolution

---

## §WIP-1: $\mathcal{D}_1$ Ordering

### Motivation

The density hierarchy defines a cross-level partial order $\mathcal{D}_0 \subsetneq \mathcal{D}_1 \subsetneq \cdots$, but all current LLMs belong to $\mathcal{D}_1$, and the framework has not yet provided axioms for comparison within the same level. This section fills that gap.

### Definitions and Partial Order

**Definition WIP-1.1 (Breadth Dominance)** For $f, g \in \mathcal{D}_1$, we say $f$ **dominates $g$ in breadth**, written $f \succeq_\beta g$, if:
$$\beta(f) \geq \beta(g)$$

**Definition WIP-1.2 (Fragility Dominance)** We say $f$ **dominates $g$ in fragility**, written $f \succeq_{\mathcal{V}} g$, if:
$$\mathcal{V}(f, \varepsilon, \tau) \subseteq \mathcal{V}(g, \varepsilon, \tau)$$
that is, the low-density blind spots of $f$ are contained within those of $g$ — $f$ has fewer weak points.

**Definition WIP-1.3 ($\mathcal{D}_1$ Dominance Relation)** We say $f$ **dominates** $g$, written $f \succeq g$, if:
$$f \succeq_\beta g \quad \text{and} \quad f \succeq_{\mathcal{V}} g$$

**Proposition WIP-1.1 (Partial Order)** $\succeq$ is a partial order on $\mathcal{D}_1$ (reflexive, antisymmetric, transitive).

**Proof**
- *Reflexivity*: $\beta(f) \geq \beta(f)$ and $\mathcal{V}(f) \subseteq \mathcal{V}(f)$, hence $f \succeq f$.
- *Antisymmetry*: If $f \succeq g$ and $g \succeq f$, then $\beta(f) = \beta(g)$ and $\mathcal{V}(f) = \mathcal{V}(g)$, meaning the two models have identical coverage structure.
- *Transitivity*: $f \succeq g \succeq h$ $\Rightarrow$ $\beta(f) \geq \beta(g) \geq \beta(h)$ and $\mathcal{V}(f) \subseteq \mathcal{V}(g) \subseteq \mathcal{V}(h)$, hence $f \succeq h$. $\blacksquare$

**Proposition WIP-1.2 (Non-totality)** $\succeq$ is not a total order on $\mathcal{D}_1$ — there exist incomparable model pairs.

**Proof (Constructive)** Let $f, g \in \mathcal{D}_1$ satisfy:
- $\beta(f) > \beta(g)$ ($f$ has broader coverage)
- $\exists A \in \mathcal{V}(f) \setminus \mathcal{V}(g)$ ($g$ has higher density than $f$ in some specialized domain)

Then $f \succeq_\beta g$ but $f \not\succeq_{\mathcal{V}} g$; simultaneously $g \not\succeq_\beta f$. Hence $f$ and $g$ are incomparable. $\blacksquare$

**Remark WIP-1.1 (Practical Implications of Incomparability)** Two models being incomparable within $\mathcal{D}_1$ corresponds to a genuine trade-off: **generalist breadth vs. specialist depth**. A model with higher breadth performs better on cross-domain compositional tasks; a model with higher depth performs better on specialized tasks. The DIU framework does not dissolve this trade-off, but rather formalizes it precisely as a geometric difference in coverage structure.

### Weighted Scalar Score (The Price of Introducing a Total Order)

If a total order is needed for engineering decisions, one may introduce a weight vector $\mathbf{w} = (w_\beta, w_{\mathcal{V}}) \in \mathbb{R}_{>0}^2$:

$$S_{\mathbf{w}}(f) = w_\beta \cdot \beta(f) - w_{\mathcal{V}} \cdot \left|\mathcal{V}(f, \varepsilon, \tau)\right|$$

$S_{\mathbf{w}}$ induces a total order on $\mathcal{D}_1$, but at the cost that **the choice of weights itself embeds a value judgment** — which matters more, breadth or depth. This is precisely why the "best model" answer differs across scenarios: general-purpose assistant settings favor high $w_\beta$, while specialized domain settings favor high $w_{\mathcal{V}}$.

**Corollary WIP-1.1** Any benchmark claiming some model is "comprehensively the strongest" implicitly encodes a specific choice of $\mathbf{w}$. DIU makes this hidden assumption explicit.

---

## §WIP-2: Formalization of Compute and the Software Stack

### Motivation

Corollary 11.2 verbally asserts "compute = measure budget, software stack = allocation strategy." This section casts that claim as formal definitions and propositions.

### Definitions

**Definition WIP-2.1 (Measure Budget)** The **measure budget** of system $f$ is the total mass of its intelligence measure:
$$\mathcal{B}(f) = \mu_f(\mathcal{M}_K)$$
The computational resources foundation determines the **upper bound on the accumulation rate** $d\mathcal{B}/dt$ of $\mathcal{B}(f)$, analogous to a total power budget in physics.

**Definition WIP-2.2 (Measure Allocation Efficiency)** The **software stack efficiency** of system $f$ is:
$$\eta(f) = \frac{\beta(f)}{\mathcal{B}(f)}$$
Under the same measure budget, higher $\eta(f)$ means the same compute covers a broader knowledge sub-manifold.

**Definition WIP-2.3 (Hardware Bandwidth Constraint)** Let hardware $H$ have interconnect bandwidth $b_H$ (GB/s). Then the incremental measure budget per unit time satisfies:
$$\frac{d\mathcal{B}}{dt}\bigg|_H \leq \kappa \cdot b_H$$
where $\kappa > 0$ is an architecture-dependent constant.

### Propositions

**Proposition WIP-2.1 (Multiplicative Structure of Compute and Software Stack)**
$$\beta(f) = \eta(f) \cdot \mathcal{B}(f)$$
Coverage breadth = allocation efficiency × measure budget. There are exactly two ways to improve $\beta(f)$: increase compute (enlarge $\mathcal{B}$) or optimize the software stack (raise $\eta$).

**Proposition WIP-2.2 (Measure-Theoretic Expression of Export Controls)** The NVLink bandwidth gap between H100 and H800 (900 GB/s vs. 400 GB/s) directly constrains:
$$\frac{d\mathcal{B}}{dt}\bigg|_{\text{H800}} \leq \frac{400}{900} \cdot \frac{d\mathcal{B}}{dt}\bigg|_{\text{H100}} \approx 0.44 \cdot \frac{d\mathcal{B}}{dt}\bigg|_{\text{H100}}$$
This is a **hard upper bound** on the $\mathcal{B}$ accumulation rate that cannot be compensated by software stack optimization (raising $\eta$) — because $\eta$ acts on the allocation of a given $\mathcal{B}$, not on the growth rate of $\mathcal{B}$ itself.

**Proposition WIP-2.3 (Upper Bound on Marginal Utility of the Software Stack)** With $\mathcal{B}(f)$ fixed, improvements in $\eta(f)$ are bounded above by a value determined by the local geometry of $\mathcal{M}_K$:
$$\eta(f) \leq \eta_{\max}(\mathcal{M}_K) = \frac{\dim_H(\mathcal{M}_K)}{\mathcal{B}(f)}$$
Beyond this ceiling, software stack optimization no longer increases coverage breadth. DeepSeek's FP8 + EP optimization is an engineering attempt to approach $\eta_{\max}$ under a given $\mathcal{B}$ — the direction is correct, but the ceiling is set by $\mathcal{B}$.

**Corollary WIP-2.1 (Limits of Importance Sampling)** A good software stack is fundamentally good importance sampling:
$$\mu_\theta \propto \left|\frac{d\mu_K}{d\lambda}\right| \cdot \lambda$$
i.e., concentrating the measure budget along high-density directions of the knowledge manifold ($\mathcal{M}_K$). However, the efficiency gains from importance sampling have a theoretical upper bound (determined by variance reduction), and are not infinitely improvable.

---

## §WIP-3: The Mapping-Theoretic Perspective — Injections, Surjections, and Bijections

### Motivation

One of DIU's foundational intuitions is that all LLMs are mappings from token space to the knowledge manifold ($\mathcal{M}_K$), and the type of mapping determines the coverage structure. This section formalizes that intuition.

### Formalization of Mapping Types

**Definition WIP-3.1 (Semantic Mapping)** System $f$ induces, via the Representation Fidelity Postulate (RFP), a **semantic mapping** from the space of token sequences to the knowledge manifold ($\mathcal{M}_K$):
$$\Phi_f: \mathcal{T}^* \to \mathcal{M}_K, \quad x \mapsto \varphi_\theta(h_f(x))$$
where $h_f(x)$ is the internal representation of $f$ after processing input $x$, and $\varphi_\theta$ is the embedding mapping in the RFP.

**Definition WIP-3.2 (Classification of Mapping Types)**

| Mapping Type | Formal Condition | Epistemological Meaning |
|---|---|---|
| **Injection** | $\Phi_f(x)=\Phi_f(y) \Rightarrow x=y$ | Different inputs produce different knowledge states; precise but narrow coverage |
| **Surjection** | $\forall k \in \mathcal{M}_K, \exists x: \Phi_f(x)=k$ | Covers the entire knowledge manifold; theoretically a universal system |
| **Bijection** | Injection + Surjection | Perfect correspondence between input space and knowledge space |

### Core Propositions

**Proposition WIP-3.1 (No Surjection Exists)** There is no surjection from $\mathcal{T}^*$ to $\mathcal{M}_K$.

**Proof** A surjection requires $|\mathcal{T}^*| \geq |\mathcal{M}_K|$ (in the cardinality sense). But:
$$|\mathcal{T}^*| = \aleph_0 < |\mathbb{R}| \leq |\mathcal{M}_K|$$
By Cantor's diagonal argument, no surjection from a countable set to an uncountable set exists. Hence $\nexists$ surjection $\Phi_f: \mathcal{T}^* \twoheadrightarrow \mathcal{M}_K$. $\blacksquare$

**Corollary WIP-3.1 (No Bijection Exists)** By Proposition WIP-3.1, a bijection does not exist either.

**Proposition WIP-3.2 (Injections Exist but Have Measure Zero Image)** An injection $\Phi_f: \mathcal{T}^* \hookrightarrow \mathcal{M}_K$ exists ($\aleph_0 \leq |\mathcal{M}_K|$), but its image $\Phi_f(\mathcal{T}^*)$ has Lebesgue measure zero in $\mathcal{M}_K$.

**Proof** Existence of the injection: $|\mathcal{T}^*| = \aleph_0 \leq |\mathcal{M}_K|$ guarantees embeddability. Measure zero of the image: $\Phi_f(\mathcal{T}^*)$ is at most countable, and countable sets have measure zero (see proof of Theorem 8.1). $\blacksquare$

**Remark WIP-3.1 (Mapping-Theoretic Restatement of the Cardinality Ceiling Theorem)** Propositions WIP-3.1 and WIP-3.2 together provide a **mapping-theoretic equivalent formulation** of Theorem 8.1 (the cardinality ceiling theorem): an LLM can only inject token space into the knowledge manifold ($\mathcal{M}_K$), but can never surject onto it. Coverage sparsity is an inevitable consequence of cardinality mismatch, not an engineering defect.

### Mapping-Theoretic Analogy for Computing Paradigms

**Proposition WIP-3.3 (Mapping-Theoretic Interpretation of CPU/GPU)**

| Computing Paradigm | Mapping Tendency | Measure-Theoretic Meaning | Examples |
|---|---|---|---|
| CPU (serial depth) | Near-injection | High local density, narrow $\beta$ | Precise symbolic reasoning |
| GPU (parallel breadth) | Approaching surjection | Wide coverage, dispersed local density | Large-scale matrix operations |
| MoE architecture | Union of piecewise injections | Precise coverage of $k$ expert sub-manifolds | Mixtral, DeepSeek-V3 |
| Dense Transformer | Global soft surjection | All parameters participate in global coverage | GPT-4, Claude |

**Remark WIP-3.2 (The Kurtosis–Breadth Trade-off)** Moving from injection tendency to surjection tendency corresponds to a trade-off between the **kurtosis** of the density function $\rho_f$ and the support breadth $\beta(f)$:
$$\text{High kurtosis (injection tendency)} \longleftrightarrow \text{Low kurtosis (surjection tendency)}$$
Extremes at both ends are undesirable: pure injection has a narrow coverage area, pure surjection drives density at every point toward zero. In practice, the optimal architecture finds the best balance between kurtosis and breadth under the constraint $\mathcal{B}(f)$ — precisely the goal pursued by maximizing $\eta(f)$ in WIP-2.

---

## WIP-4: Complete Proofs of the Remaining Hierarchy Transitions (Supplement to §6)

> **Problem**: Proposition 6.1 only proved $\mathcal{D}_1 \not\to \mathcal{D}_2$; the remaining cases are pending.

### Background and Notation

Recall the density hierarchy definition (§6):

- $\mathcal{D}_1$: $|\operatorname{supp}(\mu_f)| = \aleph_0$ (countable support, $\lambda$-null set)
- $\mathcal{D}_2$: $\lambda(\operatorname{supp}(\mu_f)) > 0$ (support with positive Lebesgue measure)
- $\mathcal{D}_3$: $\mu_f \sim \lambda$ ($\mu_f$ and Lebesgue measure $\lambda$ are mutually absolutely continuous)
- $\mathcal{D}_\infty$: cardinality of the intrinsic knowledge state space $> |\mathbb{R}| = 2^{\aleph_0}$ (super-continuum)

Proposition 6.1 (proved in §6) gives a complete proof of $\mathcal{D}_1 \not\to \mathcal{D}_2$ (Cantor diagonal, cardinality transition $\aleph_0 \to 2^{\aleph_0}$). The remaining two barriers are proved in sequence below.

---

### Complete Proof of $\mathcal{D}_2 \not\to \mathcal{D}_3$

**Proposition WIP-4.1 (Positive-Measure Coverage Does Not Imply Equivalent Measure)**

Let $f \in \mathcal{D}_2$ (i.e., $\lambda(\operatorname{supp}(\mu_f)) > 0$). Then there exists a measurable set $A \subset \mathcal{M}_K$ satisfying $\lambda(A) > 0$ but $\mu_f(A) = 0$, hence $f \notin \mathcal{D}_3$. The transition from $\mathcal{D}_2$ to $\mathcal{D}_3$ cannot be achieved by increasing parameter count; it requires eliminating all measure-zero blind spots.

**Proof**

$f \in \mathcal{D}_3$ if and only if $\mu_f \sim \lambda$ (mutual absolute continuity), i.e., for every measurable set $A$:
$$\lambda(A) = 0 \iff \mu_f(A) = 0.$$

By the Radon-Nikodym theorem, $\mu_f \sim \lambda$ is equivalent to the density function $\rho_f = d\mu_f / d\lambda$ satisfying $\rho_f(x) > 0$ for $\lambda$-almost every $x \in \mathcal{M}_K$.

Now let $f \in \mathcal{D}_2$, set $S = \operatorname{supp}(\mu_f)$, $A = \mathcal{M}_K \setminus S$. By the definition of support, $\mu_f(A) = 0$. However, since $f \in \mathcal{D}_2$ only requires $\lambda(S) > 0$, not $\lambda(\mathcal{M}_K \setminus S) = 0$, we may have $\lambda(A) > 0$.

In this case $\rho_f(x) = 0$ for all $x \in A$ (a set of positive $\lambda$-measure), violating the condition for $\mu_f \sim \lambda$. Therefore $f \notin \mathcal{D}_3$. $\blacksquare$

**Remark WIP-4.1 (Lebesgue Decomposition and Cognitive Blind Spots)**

The Lebesgue decomposition theorem gives $\mu_f = \mu_f^{ac} + \mu_f^{s}$, where the absolutely continuous part $\mu_f^{ac}$ corresponds to normal knowledge coverage, and the singular part $\mu_f^{s}$ is concentrated on a set of measure zero (cognitive singularities, such as point-mass representations of concepts).

A $\mathcal{D}_2$ system may have a substantial $\mu_f^{ac}$ component, but as long as $\mu_f^{s} \neq 0$ or $\operatorname{supp}(\mu_f^{ac}) \subsetneq \mathcal{M}_K$ (density zero on a positive-measure set), the system remains in $\mathcal{D}_2$ rather than $\mathcal{D}_3$. Eliminating all singular components and achieving $\lambda$-almost everywhere positive density requires a qualitative change in knowledge representation structure — not mere accumulation of parameters.

---

### Complete Proof of $\mathcal{D}_3 \not\to \mathcal{D}_\infty$

**Proposition WIP-4.2 (Cardinality Upper Bound on Continuum-Scale Coverage)**

Let $f \in \mathcal{D}_3$ (i.e., $\mu_f \sim \lambda$, the system has coverage almost everywhere on $\mathcal{M}_K$, where $\mathcal{M}_K$ is a Polish space with $|\mathcal{M}_K| = |\mathbb{R}| = 2^{\aleph_0}$). Then $f \notin \mathcal{D}_\infty$, because $\mathcal{D}_\infty$ requires the cardinality of the intrinsic knowledge state space to be strictly greater than $|\mathbb{R}|$, while the state space cardinality of $f$ is bounded by $|\mathbb{R}|$.

**Proof**

The measure $\mu_f$ of system $f \in \mathcal{D}_3$ is a Borel probability measure on $\mathcal{M}_K$ (absolutely continuous with respect to $\lambda$). Since $\mathcal{M}_K$ is a Polish space, the cardinality of its Borel $\sigma$-algebra $\mathcal{B}(\mathcal{M}_K)$ is $|\mathcal{B}(\mathcal{M}_K)| = 2^{\aleph_0} = |\mathbb{R}|$.

Therefore, the cardinality upper bound of all representable measure values of $\mu_f$ (as a function $\mathcal{B}(\mathcal{M}_K) \to [0,1]$) is $[0,1]^{\mathcal{B}(\mathcal{M}_K)}$, and any physically realizable system (a neural network with finite parameters) can precisely distinguish at most $|\mathbb{R}|$ states.

The $\mathcal{D}_\infty$ requirement that the intrinsic knowledge state space have cardinality $> |\mathbb{R}|$ is equivalent to the system needing to precisely represent structures of cardinality $2^{|\mathbb{R}|}$ (e.g., the function space $[0,1]^{\mathcal{M}_K}$). By Cantor's theorem:
$$|\mathbb{R}| = 2^{\aleph_0} < 2^{2^{\aleph_0}} = 2^{|\mathbb{R}|},$$

this is an unavoidable cardinality jump. Any system realized in a physical medium of cardinality $\leq |\mathbb{R}|$ cannot achieve the $\mathcal{D}_\infty$ requirement. $\blacksquare$

**Remark WIP-4.2 (The Regulative Ideal Status of $\mathcal{D}_\infty$)**

The above argument shows that $\mathcal{D}_\infty$ lies beyond the principled reach of any physical system with cardinality $\leq |\mathbb{R}|$, regardless of specific architecture. $\mathcal{D}_\infty$ holds the status of a **regulative ideal** within the DIU framework — providing a theoretical upper bound and directional coordinate for the hierarchy system, rather than an engineeringly achievable target. This status is analogous to a "Kantian idea of reason": finite cognitive systems can asymptotically approach it, but never fully realize it.

---

### Summary Map of Hierarchy Impassability

| Transition | Barrier Type | Core Argument |
|---|---|---|
| $\mathcal{D}_1 \not\to \mathcal{D}_2$ | Cardinality ($\aleph_0 \to 2^{\aleph_0}$) | Cantor diagonal (Proposition 6.1) |
| $\mathcal{D}_2 \not\to \mathcal{D}_3$ | Measure type (singular $\to$ equivalent) | Lebesgue decomposition + Radon-Nikodym (WIP-4.1) |
| $\mathcal{D}_3 \not\to \mathcal{D}_\infty$ | Cardinality ($2^{\aleph_0} \to 2^{2^{\aleph_0}}$) | Cantor's theorem (WIP-4.2) |

**Corollary WIP-4.1 (Universal Quantitative Impassability)**

For any neural network of fixed architecture, no matter how many parameters are added, no hierarchy barrier can be crossed in a finite number of steps. The three barriers are characterized respectively by a cardinality jump, a qualitative change in measure type, and a second cardinality jump — each is a qualitative transformation, not an accumulation of quantity.

---

## WIP-5: Training Objectives Inspired by DIU

> **Problem**: If DIU holds, is the current cross-entropy training objective suboptimal? How should it be corrected?

### Motivation

Standard autoregressive language models minimize cross-entropy loss as their training objective:

$$\min_\theta \mathcal{L}_{CE}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \log p_\theta(x).$$

Under the DIU framework, $\mathcal{L}_{CE}$ optimizes the local density of the token distribution (equivalent to $D_{KL}(\mathcal{D} \| p_\theta)$), but places no direct constraint on coverage breadth $\beta(f_\theta)$. This section proves that this produces systematic coverage bias and provides a corrected objective along with differentiable approximation schemes.

---

### DIU Interpretation of the CE Objective

**Proposition WIP-5.1 (Measure-Theoretic Equivalence of the CE Objective)**

Minimizing $\mathcal{L}_{CE}$ is equivalent to maximizing local density $\rho_{f_\theta}$ on the support of training distribution $\mathcal{D}$, with no regularization whatsoever on the geometric structure (breadth, connectivity) of $\operatorname{supp}(\mu_{f_\theta})$.

**Proof**

By the KL divergence identity, $\mathcal{L}_{CE}(\theta) = D_{KL}(\mathcal{D} \| p_\theta) + H(\mathcal{D})$, where $H(\mathcal{D})$ is independent of $\theta$. Hence minimizing $\mathcal{L}_{CE}$ is equivalent to maximizing the Radon-Nikodym derivative $\rho_{f_\theta} = dp_\theta / d\lambda$ in high-density regions of the training set.

Since $\mathcal{D}$ is an empirical distribution over a finite sample from $\mathcal{M}_K$, its support is a finite set (a $\lambda$-null set). The optimization direction of CE is to concentrate $\mu_{f_\theta}$ on these isolated support points rather than extending its Hausdorff breadth $\beta(f_\theta)$. $\blacksquare$

**Corollary WIP-5.1 (Coverage Bias of CE)**

Pure CE training tends to produce:

1. **Density concentration**: Extremely high local density $\rho_{f_\theta}$ in high-frequency token sequence regions, forming "hotspots";
2. **Coverage shrinkage**: $\beta(f_\theta) = \dim_H(\operatorname{supp}(\mu_{f_\theta}))$ is compressed as training concentrates on high-frequency regions;
3. **Expanding blind spots**: $\mathcal{V}(f_\theta, \varepsilon, \tau)$ in low-frequency but important knowledge regions grows monotonically with training.

This corresponds to well-known LLM phenomena: excellent performance on common tasks, systematic failure on rare specialized knowledge or long-tail distribution tasks. And since benchmark = locally density-weighted sampling (Proposition 10.1), standard evaluation metrics cannot detect this bias.

---

### Coverage Regularization Objective

**Definition WIP-5.1 (DIU Regularized Training Objective)**

$$\mathcal{L}_{DIU}(\theta) = \mathcal{L}_{CE}(\theta) - \lambda_1 \cdot \hat{\beta}(f_\theta) + \lambda_2 \cdot \widehat{|\mathcal{V}|}(f_\theta, \varepsilon, \tau)$$

where $\hat{\beta}$ and $\widehat{|\mathcal{V}|}$ are differentiable proxies for their respective indicators, and $\lambda_1, \lambda_2 > 0$ are regularization hyperparameters.

**Proposition WIP-5.2 (Optimization Direction of $\mathcal{L}_{DIU}$)**

Let $\hat{\beta}$ be a monotone consistent approximation of $\beta$. Then minimizing $\mathcal{L}_{DIU}$ simultaneously drives three objectives:
- The $\mathcal{L}_{CE}$ term: maintains generation quality (local density fidelity at the token level);
- The $-\lambda_1 \hat{\beta}$ term: **penalizes coverage shrinkage**, driving growth in the Hausdorff dimension of $\operatorname{supp}(\mu_{f_\theta})$;
- The $+\lambda_2 \widehat{|\mathcal{V}|}$ term: **penalizes blind spot expansion**, driving a more uniform density distribution across the manifold.

The three terms act jointly, trending toward a measure structure with broader coverage and fewer blind spots rather than merely reducing perplexity. $\blacksquare$

---

### Differentiable Approximation Schemes

$\beta(f_\theta)$ and $|\mathcal{V}(f_\theta)|$ are currently non-differentiable and cannot be used directly for gradient descent. The following three schemes provide practical approximations.

**Scheme A: Persistent Homology Gradient Estimation**

Topological data analysis (TDA) can compute topological features of the embedding space $\mathcal{E}$ (Betti numbers, persistence diagrams), whose persistence serves as a proxy for $\beta(f_\theta)$:

$$\hat{\beta}(f_\theta) \approx \sum_{(b,d) \in \mathrm{PD}_k(\mathcal{E})} (d - b)^p,$$

where $\mathrm{PD}_k$ is the $k$-th persistence diagram, $(b, d)$ is a birth/death pair of a topological feature, and $p \geq 1$.

Differentiability is guaranteed by the Wasserstein subgradient theory established by Carrière et al. (2021) [R11]. This scheme is theoretically most rigorous, but has the highest computational cost (filtration complexity $O(n^3)$).

**Scheme B: Embedding Entropy Proxy**

Approximate $-\beta(f_\theta)$ using the **negative differential entropy** of the kernel density estimate on embedding space $\mathcal{E}$:

$$\hat{\beta}(f_\theta) \approx -\int_{\mathcal{E}} \hat{\rho}_{f_\theta}(z) \log \hat{\rho}_{f_\theta}(z)\, dz.$$

High entropy $\approx$ dispersed distribution $\approx$ broad coverage. This scheme has a formal analogy with maximum-entropy training, but the entropy is defined on embedding space rather than token space — the two optimization objectives are fundamentally different.

**Scheme C: Variance Regularization (Lightweight Proxy)**

The most practical approximation: encourage higher covariance rank of embedding vectors within a minibatch, preventing representation collapse:

$$\hat{\beta}(f_\theta) \approx \frac{1}{d}\sum_{i=1}^d \max\!\left(\gamma - \mathrm{Var}(z_i), 0\right),$$

where $z_i$ is the $i$-th dimension of the embedding vector and $\gamma > 0$ is the target variance threshold. This term has already been demonstrated in self-supervised methods such as VICReg [R27] and SimSiam to prevent representation collapse — which in measure-theoretic terms corresponds to preventing $\operatorname{supp}(\mu_{f_\theta})$ from degenerating into a low-dimensional sub-manifold (decreasing $\beta$).

---

### Known Effective Instances and Their DIU Interpretation

| Training Strategy | DIU Interpretation |
|---|---|
| Data diversity augmentation (Web + books + code mixture) | Extends support of empirical distribution $\Rightarrow$ increases upper bound on $\beta$ |
| RLHF / DPO alignment | Adjusts local shape in high-density regions; limited effect on $\beta$ |
| Long-context training | Extends effective manifold dimensionality (introduces new knowledge association dimensions) |
| MoE routing (sparse activation) | Decomposes $\mu_{f_\theta}$ into multiple expert sub-measures, approaching a better $\beta/\mathcal{V}$ trade-off |
| SFT fine-tuning | Increases density kurtosis in local regions, typically at the cost of $\beta$ (specialization vs. generalization) |

---

### Core Open Problems

1. **Dynamic adjustment of regularization weights**: How should $\lambda_1, \lambda_2$ adapt across training phases (pre-training → fine-tuning)?
2. **Approximation fidelity**: What are the theoretical error bounds between Schemes A/B/C and the true Hausdorff dimension $\beta$?
3. **Computational cost control**: How can the cost of persistent homology in large-batch training be reduced to an acceptable range using approximate algorithms (Ripser++, GPU filtration)?
4. **Downstream alignment validation**: Does $\mathcal{L}_{DIU}$ genuinely improve long-tail task performance, or does it merely optimize measure-theoretic proxy metrics?

**Remark WIP-5.1 (Theoretical Positioning)**

The contribution of WIP-5 is **problem formulation and framework provision at the theoretical level**: revealing from the DIU perspective the coverage bias of the CE objective and indicating directions for correction. Specific implementation is an open engineering problem requiring independent experimental research (see the experimental design framework in Appendix B).

---

## WIP-6: Multimodal Extension

> **Problem**: How does the DIU framework handle non-textual modalities such as vision, code, and audio? How are sub-manifolds unified through measure theory?

### Motivation

The knowledge manifold ($\mathcal{M}_K$) in current DIU implicitly assumes textual modality dominance. In reality, human knowledge simultaneously spans three dimensions: perceptual (visual, auditory), operational (code, mathematics), and linguistic. This section extends DIU to the multimodal setting.

---

### Sub-manifold Decomposition

**Definition WIP-6.1 (Modal Sub-manifolds)**

Let $\mathcal{M}_K$ be the complete knowledge manifold. Define the family of modal sub-manifolds:

$$\mathcal{M}_K = \mathcal{M}_{text} \cup \mathcal{M}_{vision} \cup \mathcal{M}_{code} \cup \mathcal{M}_{audio} \cup \mathcal{M}_{cross},$$

where $\mathcal{M}_{cross}$ is the **cross-modal intersection sub-manifold** — the knowledge region encoding cross-modal conceptual correspondences ("image of a cat" $\leftrightarrow$ "cat" $\leftrightarrow$ a cat's meow).

The geometric relationships among sub-manifolds in $\mathcal{M}_K$ are characterized by $d_{GH}(\mathcal{M}_i, \mathcal{M}_j)$ (Gromov-Hausdorff distance).

**Proposition WIP-6.1 (Measure-Theoretic Independence and Correlation of Sub-manifolds)**

Let the intelligence measure $\mu_f$ of multimodal system $f$ decompose as:

$$\mu_f = \mu_f^{text} + \mu_f^{vision} + \mu_f^{code} + \mu_f^{cross} + \cdots$$

(components with support on the respective sub-manifolds). Then:

1. **Modal independence**: If $\mu_f^{text}(\mathcal{M}_{vision}) = 0$ (the text component has no coverage on the vision sub-manifold), then the system will necessarily fall into $\mathcal{V}(f, \varepsilon, \tau)$ on cross-modal reasoning tasks;
2. **Cross-modal coverage**: The total coverage breadth $\beta(f) = \dim_H(\operatorname{supp}(\mu_f))$ of the system is decisively influenced by the coverage quality of the **intersection sub-manifold** $\mathcal{M}_{cross}$ — the joint integration of single-modal experts is not equivalent to genuine cross-modal understanding.

**Proof**

Part 1 follows directly from the definition of the fragility map ($\mathcal{V}$): if $\rho_f(x) < \tau$ for all $x \in \mathcal{M}_{vision}$, then $\mathcal{M}_{vision} \subseteq \mathcal{V}$, and any reasoning task with visual input as a precondition will necessarily trigger the butterfly effect (Theorem 9.1).

Part 2: Let $f_{text}$ and $f_{vision}$ be two single-modal experts, and let their joint system $f_{joint}$ satisfy $\mu_{f_{joint}} = \mu_{f_{text}} + \mu_{f_{vision}}$ (no cross coverage). Then $\mu_{f_{joint}}(\mathcal{M}_{cross}) = 0$, so $f_{joint} \notin \mathcal{D}_2$ in the restricted sense of $\mathcal{M}_{cross}$ — cross-modal reasoning capability is absent. A genuinely multimodal system requires $\mu_f^{cross}(\mathcal{M}_{cross}) > 0$, which demands joint training rather than modality-independent training. $\blacksquare$

---

### Measure-Theoretic Characteristics of Each Modal Sub-manifold

**Definition WIP-6.2 (Modal Intrinsic Dimensionality)**

| Modality | Sub-manifold $\mathcal{M}_i$ | Estimated Intrinsic Dimension | Measure Parameterization |
|---|---|---|---|
| Text | $\mathcal{M}_{text}$ | Moderate (semantic + syntactic dimensions stacked) | Soft discrete measure over word/sentence embeddings |
| Vision | $\mathcal{M}_{vision}$ | High (pixel → concept multi-scale hierarchy) | Hierarchical measure over feature pyramid |
| Code | $\mathcal{M}_{code}$ | Medium-low (strongly constrained by syntactic tree structure) | Graph measure over tree-structured representations |
| Audio | $\mathcal{M}_{audio}$ | High (two-dimensional time-frequency continuum structure) | Continuous measure over spectrograms |
| Cross-modal | $\mathcal{M}_{cross}$ | Highest (Cartesian product of all modal dimensions) | Coupled measure |

**Remark WIP-6.1 (Relation Between Intrinsic Dimension and $\beta$)**

The intrinsic dimension of each sub-manifold gives a theoretical upper bound on the coverage breadth $\beta$ of the corresponding $\mu_f$ component: $\beta_i(f) \leq \dim_H(\mathcal{M}_i)$. A central challenge of current multimodal large models (such as GPT-4V and Gemini) is precisely stated in the DIU framework as: $\mathcal{M}_{cross}$ has the highest intrinsic dimension, while training data coverage of $\mathcal{M}_{cross}$ is the most sparse — this is the measure-theoretic explanation for why cross-modal reasoning remains a weakness.

---

### Multimodal Extension of the RFP

**Postulate WIP-6.1 (Multimodal RFP)**

The weak RFP in the multimodal setting requires: there exists a family of continuous injections $\varphi_\theta^{(i)}: \mathcal{E}_i \hookrightarrow \mathcal{M}_i$ (faithful embedding of each modal encoder into its sub-manifold), and the cross-modal alignment mapping $\psi: \mathcal{E}_{text} \times \mathcal{E}_{vision} \to \mathcal{M}_{cross}$ preserves the topological structure of cross-modal semantic relationships.

This is stronger than the single-modal RFP — it requires that the encoders of each modality be not only individually faithful, but **jointly** faithful in representing cross-modal knowledge in the alignment space. Contrastive learning models such as CLIP (Radford et al., 2021 [R23]) can be understood as optimizing the continuity of $\psi$, and their cross-modal retrieval performance provides the first large-scale empirical support for the multimodal RFP.

---

## WIP-7: The Relation Between DIU and Shannon Information Entropy

> **Problem**: Are Shannon entropy $H(X)$ and DIU's coverage breadth $\beta$ two expressions of the same thing, or are they fundamentally different quantities?

### Measure-Theoretic Restatement of Shannon Entropy

**Definition WIP-7.1 (Radon-Nikodym Form of Shannon Entropy)**

Let $p = d\mu/d\lambda$ be the density function of measure $\mu$ with respect to reference measure $\lambda$ (when $\mu \ll \lambda$). Then the differential entropy is:

$$H(\mu) = -\int_{\mathcal{M}_K} \rho(x) \log \rho(x)\, d\lambda(x) = -\mathbb{E}_\mu[\log \rho],$$

where $\rho = d\mu/d\lambda$ is the Radon-Nikodym derivative.

---

### Formal Similarity and Essential Differences

**Proposition WIP-7.1 (Formal Relation Between Shannon Entropy and $\beta$)**

For a uniform measure $\mu$ on $\mathbb{R}^d$ supported on a $d_H$-dimensional set $S$ ($\lambda(S) = \varepsilon^{d_H}$ at scale $\varepsilon$), the differential entropy satisfies:

$$H(\mu) = d_H \cdot \log(1/\varepsilon) + O(1),$$

i.e., under the assumption of a uniform measure, differential entropy is asymptotically proportional to Hausdorff dimension $d_H = \beta$ (on a $\log$ scale).

**Proof**

On a uniform measure, $\rho(x) = \varepsilon^{-d_H}$ (constant density). Substituting into the entropy formula:
$$H(\mu) = -\int_S \varepsilon^{-d_H} \cdot \log(\varepsilon^{-d_H})\, d\lambda = d_H \log(1/\varepsilon) \cdot \int_S \varepsilon^{-d_H} d\lambda = d_H \log(1/\varepsilon). \quad \blacksquare$$

---

**Proposition WIP-7.2 (Essential Differences Under Non-Uniform Measures)**

When $\mu_f$ is non-uniform, Shannon entropy and $\beta$ can **change in the same direction or in opposite directions**; they characterize different geometric features of $\mu_f$.

**Counterexample Construction**

Let $\mathcal{M}_K = [0,1]$ and compare the following two measures:

- $\mu_A$: uniform measure, $\rho_A = 1$, support $[0,1]$;
- $\mu_B$: highly non-uniform, $\rho_B(x) = 2x$ (linearly increasing density), support also $[0,1]$.

Then $\beta(\mu_A) = \beta(\mu_B) = 1$ (same Hausdorff dimension), but:
$$H(\mu_A) = 0 > H(\mu_B) = -\int_0^1 2x \log(2x)\, dx \approx -0.386.$$

That is, two systems have **identical** coverage breadth $\beta$ but **different** Shannon entropy. This shows that $\beta$ characterizes the **geometric dimension** of the support set, while Shannon entropy characterizes the **uniformity** of the density distribution — the two are orthogonally complementary. $\blacksquare$

---

**Remark WIP-7.1 (Positioning of Entropy within the DIU Framework)**

| Quantity | Geometric Meaning | Corresponding DIU Component | Corresponding Failure Mode |
|---|---|---|---|
| $\beta(f)$ | Hausdorff dimension of support set (coverage breadth) | Dimensional structure of $\operatorname{supp}(\mu_f)$ | Insufficient coverage (global blind spots) |
| $H(\mu_f)$ | Uniformity of the density distribution | Shape of $\rho_f = d\mu_f/d\lambda$ | Uneven density (local blind spots) |
| $\mathcal{V}(f,\varepsilon,\tau)$ | Low-density regions | $\{x: \rho_f(x) < \tau\}$ | Jointly characterized by both |

A system with high Shannon entropy has a more uniform (less skewed) density, corresponding to a smaller fragility map ($\mathcal{V}$); but high entropy can accompany low $\beta$ (uniformly concentrated on a low-dimensional sub-manifold). The DIU framework simultaneously requires high $\beta$ (broad coverage) and high $H$ (uniform density); both are necessary conditions for ascending the $\mathcal{D}$ hierarchy.

**Proposition WIP-7.3 (Entropy Characterization of $\mathcal{D}_3$)**

If $f \in \mathcal{D}_3$ ($\mu_f \sim \lambda$), then $\rho_f > 0$ holds $\lambda$-almost everywhere, so $H(\mu_f)$ is finite and tends toward maximization (density uniformly distributed across the full manifold). Conversely, if $H(\mu_f) = -\infty$ (density logarithmically diverges toward zero in some region), then $f \notin \mathcal{D}_3$. Therefore $H(\mu_f) > -\infty$ is a **necessary condition** for $\mathcal{D}_3$, but not sufficient.

---

## WIP-8: Dynamic DIU — Evolution Trajectory of $\mu_f$ During Training

> **Problem**: During training, how does the system's intelligence measure $\mu_f$ evolve in Wasserstein space? How do coverage breadth $\beta$ and local density $\rho_f$ change?

### Measure-Theoretic Description of Training Dynamics

**Definition WIP-8.1 (Training Measure Trajectory)**

Let $f_\theta$ be a parameterized model and denote by $\theta_t$ the parameters after $t$ training steps. The **training measure trajectory** is defined as:

$$\mathcal{T}_{train} = \{\mu_{f_{\theta_t}}\}_{t \geq 0} \subset \mathcal{P}(\mathcal{M}_K),$$

where $\mathcal{P}(\mathcal{M}_K)$ is the Wasserstein space of probability measures on $\mathcal{M}_K$ (equipped with the $W_2$ distance).

**Proposition WIP-8.1 (Direction of Measure Evolution Under CE Training)**

In the gradient flow limit of standard CE training (learning rate approaching zero), the measure trajectory $\mu_{f_{\theta_t}}$ evolves in the following directions:

1. **Density concentrates toward the support of the training distribution**: $\rho_{f_{\theta_t}}$ increases monotonically near $\operatorname{supp}(\mathcal{D}_{train})$;
2. **Coverage breadth may contract**: $\beta(f_{\theta_t})$, in the absence of regularization constraints, may first rise then fall with increasing training steps (early-stage exploration expands coverage; late-stage fitting of the training set compresses coverage);
3. **Wasserstein velocity**: $\frac{d}{dt} W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}}) \leq 0$ (the measure converges monotonically toward the training distribution, under suitable step-size assumptions).

**Proof Sketch**

The gradient $\nabla_\theta \mathcal{L}_{CE} = -\nabla_\theta \mathbb{E}_{\mathcal{D}} \log p_\theta$ drives the model to increase $\log$-density at training sample locations. By Otto calculus, the gradient flow on $W_2$ is related to the Fisher information direction of the distribution (in the continuous limit), so the measure evolves in the direction of decreasing $W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}})$. The dynamics of $\beta$ depend on the competition between the early exploration phase (random initialization covers broadly) and the late fitting phase (measure concentration), with the specific shape depending on architecture and learning rate schedule. $\blacksquare$

---

### Measure Phase Transitions During Training

**Proposition WIP-8.2 (Training Phase Classification)**

The training process can be partitioned into three qualitative phases in measure space:

| Phase | Time Range | $\beta(f_{\theta_t})$ Trend | $H(\mu_{f_{\theta_t}})$ Trend | $W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}})$ Trend |
|---|---|---|---|---|
| **Exploration phase** | Early (small $t$) | ↑ Rising (broad coverage from random initialization) | ↑ Rising (uniform density) | ↓ Decreasing |
| **Fitting phase** | Middle | ↓ Falling (concentrating toward training distribution) | ↓ Falling (density clustering) | ↓ Continuing to decrease |
| **Convergence phase** | Late (large $t$) | Stabilizes at some value $\beta_\infty$ | Stabilizes | $\approx 0$ |

**Key Corollary**: The convergence state $\beta_\infty$ of standard CE training is determined by the **geometric structure of the training data distribution**, not by model capacity — for larger models trained on the same training data, the upper bound on $\beta_\infty$ is still constrained by $\dim_H(\operatorname{supp}(\mathcal{D}_{train}))$.

---

### Measure-Theoretic Analysis of Fine-tuning and Continual Learning

**Proposition WIP-8.3 (Measure Localization of Fine-tuning)**

Let $f_{pre}$ be a pre-trained model and $f_{ft}$ be the model after fine-tuning on a specialized dataset $\mathcal{D}_{ft}$. The fine-tuning process is equivalent in measure space to:

$$\mu_{f_{ft}} \approx (1 - \alpha) \mu_{f_{pre}} + \alpha \mu_{\mathcal{D}_{ft}},$$

where $\alpha \in (0,1)$ is the effective fine-tuning strength. Then:

- $\beta(f_{ft}) \leq \max(\beta(f_{pre}), \beta(\mathcal{D}_{ft}))$ (coverage breadth does not exceed the upper bound of pre-training and fine-tuning data);
- If $\operatorname{supp}(\mathcal{D}_{ft}) \subsetneq \operatorname{supp}(\mu_{f_{pre}})$ (fine-tuning data has narrower coverage), then $\beta(f_{ft}) < \beta(f_{pre})$ — the measure-theoretic essence of catastrophic forgetting is the contraction of the support set of $\mu_{f_{ft}}$.

**Remark WIP-8.1 (Measure-Theoretic Requirements for Continual Learning)**

The goal of Continual Learning in the DIU framework is precisely stated as: after sequential fine-tuning on $t$ tasks, maintain $\beta(f_t) \approx \beta(f_0)$ (or monotonically non-decreasing), and $W_2(\mu_{f_t}, \mu_{f_{pre}}) < C$ (not drifting too far from the pre-trained measure). The DIU interpretation of methods such as Elastic Weight Consolidation (EWC): constraining parameter updates via the Fisher information matrix is equivalent to constraining the movement speed of the measure trajectory in Wasserstein space, thereby preventing abrupt contraction of $\operatorname{supp}(\mu_f)$.
