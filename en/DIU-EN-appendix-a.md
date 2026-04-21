# DIU Appendix A: Formalization of Core Structural Diagrams

---

## Appendix A.1: Geometric Illustration of the Knowledge Manifold

### A.1.1 Knowledge Manifold and Intelligence Measure

The following diagram describes the basic geometric structure of the measure $\mu_f$ induced by system $f$ on the knowledge manifold ($\mathcal{M}_K$):

```
  Knowledge manifold 𝓜_K (complete separable metric space)
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ████████████████                                   │
  │  █ supp(μ_f) █     ░░░░░░░░░░░░                    │
  │  █ high-density █  ░ 𝒱(f,ε,τ) ░  ← fragility map  │
  │  ████████████████  ░ (low-density)░                 │
  │        ↑           ░░░░░░░░░░░░                    │
  │   ρ_f(x) = dμ_f/dλ              ···· uncovered region │
  │   (Radon-Nikodym derivative)     ···· μ_f = 0      │
  │                                                     │
  └─────────────────────────────────────────────────────┘

  β(f) = dim_H(supp(μ_f))     ← coverage breadth (Hausdorff dimension)
  Δ(f_A, f_B) = W₂(μ_A, μ_B)  ← structural distance between models
```

**Formal Correspondence**

| Geometric Region | Measure-Theoretic Characterization | Practical Interpretation |
|---|---|---|
| High-density core | $\{x : \rho_f(x) \geq \rho_{\max}/2\}$ | Domain of competence (captured by benchmark) |
| Normal coverage zone | $\{x : \rho_f(x) \in [\tau, \rho_{\max}/2)\}$ | Usable but unstable region |
| Fragility map ($\mathcal{V}$) | $\{x : \rho_f(x) < \tau\}$ | Region triggering the butterfly effect |
| Zero-coverage zone | $\mathcal{M}_K \setminus \operatorname{supp}(\mu_f)$ | Complete blind spot |

---

### A.1.2 Modal Sub-manifold Decomposition (Multimodal Extension)

```
         Knowledge manifold 𝓜_K
         ┌──────────────────────────────┐
         │  𝓜_text   𝓜_vision          │
         │  ┌──────┐  ┌──────┐         │
         │  │ ████ │  │ ████ │         │
         │  └──┬───┘  └───┬──┘         │
         │     └────┬─────┘            │
         │     𝓜_cross (cross-modal)   │
         │       ┌──────┐              │
         │       │ ████ │ ← sparse coverage │
         │       └──────┘              │
         │  𝓜_code    𝓜_audio          │
         │  ┌──────┐  ┌──────┐         │
         │  │ ████ │  │ ████ │         │
         │  └──────┘  └──────┘         │
         └──────────────────────────────┘
```

$\mathcal{M}_{cross}$ has the highest intrinsic dimensionality yet the sparsest training coverage — this constitutes the measure-theoretic bottleneck of cross-modal reasoning in current multimodal large language models.

---

## Appendix A.2: Partial Order Structure of the Density Hierarchy

### A.2.1 Hierarchical Partial Order Diagram

```
  𝒟_∞   super-continuum         |𝓜_f| > |ℝ|
    ↑
  ══════════════════  Cantor theorem barrier
  (2^|ℝ| vs |ℝ|)     principled upper bound for physical systems
    ↑
  𝒟_3   continuum-dense         μ_f ∼ λ (equivalent measures)
    ↑
  ══════════════════  Lebesgue decomposition barrier
  (singular → equivalent)  elimination of all measure-zero blind spots
    ↑
  𝒟_2   positive measure coverage   λ(supp(μ_f)) > 0
    ↑
  ══════════════════  cardinality ceiling  ← all current LLMs lie below this
  (ℵ₀ → 2^ℵ₀)        Cantor diagonal argument
    ↑
  𝒟_1   countably dense         |supp(μ_f)| = ℵ₀   ← GPT-4, Claude, ...
    ↑
  𝒟_0   finite coverage         |supp(μ_f)| < ∞    ← rule systems, lookup tables
```

### A.2.2 Overview of Barrier Types

```
Transition         Obstacle         Proof Tool                Required Qualitative Change
────────────────────────────────────────────────────────────────
𝒟_0 → 𝒟_1       cardinality ℵ₀    — (finite → countable)      countably infinite symbol system
𝒟_1 → 𝒟_2       cardinality 2^ℵ₀  Cantor diagonal             uncountable representational basis
𝒟_2 → 𝒟_3       measure type      Lebesgue decomp. + R-N      almost everywhere coverage of full manifold
𝒟_3 → 𝒟_∞       cardinality 2^|ℝ| Cantor theorem              super-continuum physical medium
```

---

## Appendix A.3: Architecture Unification Commutative Diagram

### A.3.1 Measure Parameterization of Different Architectures

The following diagram illustrates the architecture reduction relations of Theorem 7.1:

```
                    DIU unified representation
              f_θ(x) = ∫ k_θ(x,x') v(x') dμ_θ(x')
                           ↑
              ┌────────────┼────────────┐
              │            │            │
        Transformer    Mamba/SSM    World Model
        ─────────────  ──────────   ───────────
        k_θ = softmax  k_θ = e^A(t-s)B  k_θ = transition kernel
        μ_θ = Σ αᵢδᵥᵢ  μ_θ = continuous kernel  μ_θ = path measure
        (discrete measure) (adaptive continuous)  (Markov chain)
              │            │            │
              └────────────┴────────────┘
                   Differences reside solely in:
                   choice of base space 𝒳
                   choice of measure type
                   form of kernel function k_θ
```

### A.3.2 Parameterization Chain

$$\theta \xrightarrow{\text{forward pass}} f_\theta \xrightarrow{\text{output distribution}} \mu_{f_\theta} \xrightarrow{\text{supp}} \beta(f_\theta), \mathcal{V}(f_\theta) \xrightarrow{\text{evaluation}} S_{\mathbf{w}}(f_\theta)$$

---

## Appendix A.4: Wasserstein Distance and Model Comparison

### A.4.1 Structural Distance Between Models

```
  Measure space 𝒫(𝓜_K) (equipped with W₂ metric)

        μ_A              μ_B
  ┌──────────┐      ┌──────────┐
  │  ████    │      │   ████   │
  │ ██████   │      │  █████   │
  │  ████    │      │   ████   │
  └──────────┘      └──────────┘
       │                 │
       └────────┬─────────┘
                │
          W₂(μ_A, μ_B)
          = minimum transport cost
          = "minimum work required to reshape A's knowledge coverage into B's"
```

**W₂ vs. KL Divergence (Illustration of Proposition 13.2)**

| Scenario | KL(μ_A ‖ μ_B) | W₂(μ_A, μ_B) |
|---|---|---|
| Disjoint supports (completely non-overlapping coverage) | $+\infty$ | Finite (still comparable) |
| Same mean but different distributional shapes | Finite | Finite |
| One model has systematic shift in a domain | Insensitive | Captures displacement |

$W_2$ is more robust than KL divergence for comparing coverage structures; this is the geometric rationale for DIU's choice of $W_2$ as its core measure.

---

## Appendix A.5: Training Measure Trajectory (Dynamic DIU)

### A.5.1 Three-Phase Evolution under CE Training

```
  Coverage breadth β(f_θt)
  │
  │     Exploration phase   Fitting phase     Convergence phase
  │   ╱──────╲             ╲               ─────────  β∞
  │ ╱          ╲             ╲─────────────╱
  │╱             ╲
  │────────────────────────────────────────────────── training step t
  │               ↑
  │           "breadth peak"
  │           (early stopping point = DIU optimal stopping point)

  W₂(μ_θt, μ_𝒟train)
  │╲
  │  ╲
  │    ╲──────────────────────────────────── → 0 convergence
  │────────────────────────────────────────── training step t
```

**Key Corollary**: $\beta_\infty$ is determined by the geometric structure of the training data, not by the number of model parameters. Early stopping, from the DIU perspective, is equivalent to **stopping at the breadth peak to preserve a broader coverage structure**.

---

## Appendix A.6: Epistemological Chain of the RFP

```
Observed embedding space 𝓔          Knowledge manifold 𝓜_K
                                       (not directly observable)
  ┌──────────────┐                         ┌──────────────┐
  │ linear semantic algebra │ ─────φ_θ──→  │ locally linear coordinates │
  │ cross-lingual alignment │ ─────φ_θ──→  │ language-agnostic structure │
  │ probe linear separability │ ───φ_θ──→  │ attribute coordinate encoding │
  │ interpolation semantic continuity │ ──φ_θ──→ │ path connectivity │
  └──────────────┘                         └──────────────┘
         ↓                                        ↓
    Weak RFP holds                         GH convergence d_GH→0
  (topologically faithful, empirically supported)
                                      Strong RFP: open problem
                                      (measure-preserving, no validation method)
```

---
