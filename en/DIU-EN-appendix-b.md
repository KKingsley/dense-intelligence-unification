# DIU Appendix B: Experimental Design

---

## Appendix B: Experimental Design for Open Problem 2

### B.1 Core Hypotheses

**H1 (Breadth Predictive Power)** The coverage breadth ($\beta$) $\beta(f)$ exhibits greater predictive power for performance on complex multi-step tasks than any finite benchmark combination.

**H2 (Fragility Map Predictive Power)** The fragility map ($\mathcal{V}$) $\mathcal{V}(f,\varepsilon,\tau)$ can predict specific failure locations before task failures occur.

**H3 (Structural Distance Superiority)** $W_2(\mu_A,\mu_B)$ provides superior explanatory power for inter-model performance gaps compared to task-specific score differences.

### B.2 Query Set Construction

```
|Q| ≥ 10,000, stratified sampling along the following dimensions:

Domain dimension (~15% each): mathematics / code / scientific reasoning /
                               language understanding / cross-domain composition /
                               commonsense / metacognition

Complexity dimension:
  L1 — single-step knowledge retrieval     (capturable by benchmark)
  L2 — two-step causal reasoning
  L3 — multi-step cross-domain composition (not capturable by benchmark)
  L4 — adversarial perturbation tasks
```

Key design principle: **the reasoning path of L3/L4 tasks must traverse the transition regions of sub-manifolds**, which constitutes the induction zone of $\mathcal{V}(f)$.

### B.3 Metric Computation Pipeline

**Step 1: Embedding Collection**
Fix a reference encoder (E5-mistral-7b); collect $E_f = \{\phi(\text{response}_f(q_i))\}_{i=1}^N \subset \mathbb{R}^d$

**Step 2: Coverage Breadth $\beta(f)$**
$$\hat\beta(f) = \widehat{\dim}_{\text{TwoNN}}(E_f) = \left(\frac{1}{N}\sum_{i=1}^N \log\frac{r_{i,2}}{r_{i,1}}\right)^{-1}$$

**Step 3: Structural Distance $W_2$**
$$\widetilde W_2(\hat\mu_A,\hat\mu_B) = \left(\int_{\mathbb{S}^{d-1}} W_2^2(P_\theta\hat\mu_A, P_\theta\hat\mu_B)\,d\theta\right)^{1/2}$$
Monte Carlo integral approximation (~1000 random projections)

**Step 4: Fragility Map $\mathcal{V}$**
$$\hat\rho_f(x_i,k) = \frac{k}{N \cdot V_d \cdot r_{i,k}^d}$$
Label $\hat{\mathcal{V}}(f,k,\tau) = \{x_i : \hat\rho_f(x_i,k) < \tau\}$, clustered by semantic category

### B.4 Validation Protocol

**Experiment 1: Correlation Between Coverage Breadth and Complex Task Performance**

| Measurement | Expected Result |
|---|---|
| $\text{Corr}(\beta(f),\; \text{perf}_{L3})$ | $> 0.7$ |
| $\text{Corr}(\text{MMLU},\; \text{perf}_{L3})$ | $< 0.55$ |
| $\text{Corr}(\beta(f),\; \text{perf}_{L1})$ | $\approx \text{Corr}(\text{MMLU},\; \text{perf}_{L1})$ |

**Experiment 2: Fragility Map Prediction of Failure Modes**

1. Compute $\hat{\mathcal{V}}(f)$ for each model; identify low-density semantic clusters
2. Design a targeted probe task set $\mathcal{Q}_{\mathcal{V}}$: query points fall within the $\hat{\mathcal{V}}$ region
3. Blind evaluation: predict failure rates based on $\hat{\mathcal{V}}$ prior to observing scores
4. Validate the Lyapunov amplification coefficient $\Lambda \approx \log(1/\hat\tau)$

**Experiment 3: DIU Detection of Benchmark Contamination**

$$\text{Contamination Score}(f,\mathcal{B}) = \frac{\hat\rho_f(x_{\mathcal{B}},\varepsilon)}{\beta(f)} - 1$$

Significantly positive → artificially elevated local density in that region → signal of benchmark contamination

### B.5 Experimental Resources

```
Models: GPT-4o, Claude-3.5-Sonnet, Llama-3-70B,
        DeepSeek-V3, Qwen-2.5-72B, Mistral-Large

Query set: 10k entries, annotated with domain + complexity level
Reference encoder: E5-mistral-7b (open-source, reproducible)
Compute: embeddings ~6 GPU-hours; TwoNN/SW distance ~2h CPU
```

### B.6 Falsifiability Conditions

If any of the following holds, the core claims of DIU require revision:

- $\text{Corr}(\beta(f), \text{perf}_{L3}) < 0.4$ (coverage breadth has no predictive power)
- $\hat{\mathcal{V}}$ prediction accuracy for failure locations does not exceed random baseline
- $\text{Corr}(\text{MMLU}, \text{perf}_{L3}) > 0.85$ (benchmark is already sufficient)
