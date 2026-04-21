# DIU §2: Mathematical Preliminaries

---

## §2 Mathematical Preliminaries

### 2.1 Measure Spaces

**Definition 2.1 (Measure Space)** A triple $(\Omega, \Sigma, \mu)$ is called a measure space, where $\Omega$ is the sample space, $\Sigma$ is a $\sigma$-algebra, and $\mu: \Sigma \to [0,+\infty]$ satisfies countable additivity.

**Definition 2.2 (Absolute Continuity)** Let $\mu, \lambda$ be two measures on $(\Omega, \Sigma)$. If for every $A \in \Sigma$, $\lambda(A)=0 \Rightarrow \mu(A)=0$, then $\mu$ is said to be **absolutely continuous** with respect to $\lambda$, written $\mu \ll \lambda$.

**Theorem 2.1 (Radon-Nikodym)** If $\mu \ll \lambda$ and $\lambda$ is a $\sigma$-finite measure, then there exists a unique ($\lambda$-a.e.) non-negative measurable function $\rho = d\mu/d\lambda$ such that:
$$\mu(A) = \int_A \rho \, d\lambda, \quad \forall A \in \Sigma$$
$\rho$ is called the **density function**, and serves as the formal object representing "local density" in DIU.

### 2.2 Hausdorff Dimension

**Definition 2.3 (Hausdorff Measure)** For a set $E$ in a metric space $(X,d)$, the $s$-dimensional Hausdorff measure is:
$$\mathcal{H}^s(E) = \lim_{\delta \to 0} \inf\left\{\sum_i \operatorname{diam}(U_i)^s : E \subseteq \bigcup_i U_i,\; \operatorname{diam}(U_i) < \delta\right\}$$

**Definition 2.4 (Hausdorff Dimension)**
$$\dim_H(E) = \inf\{s \geq 0 : \mathcal{H}^s(E) = 0\} = \sup\{s \geq 0 : \mathcal{H}^s(E) = \infty\}$$
This characterizes the "fractal breadth" of covering a set—used in DIU to measure the structural width of intelligence coverage.

### 2.3 Wasserstein Distance

**Definition 2.5 ($p$-Wasserstein Distance)** Let $\mu, \nu$ be probability measures on a metric space $(X,d)$, and let $\Gamma(\mu,\nu)$ denote the set of all joint distributions with marginals $\mu$ and $\nu$ respectively:
$$W_p(\mu,\nu) = \left(\inf_{\gamma \in \Gamma(\mu,\nu)} \int_{X \times X} d(x,y)^p \, d\gamma(x,y)\right)^{1/p}$$

**Remark 2.1** The geometric interpretation of the $W_2$ distance: the minimum cost of transporting probability mass from $\mu$ to $\nu$ (measured in squared Euclidean distance). $W_2$ endows the space of probability measures with a Riemannian manifold structure (Otto calculus), and is used in DIU to compare the overall difference between the coverage structures of two models.

### 2.4 Gromov-Hausdorff Distance

**Definition 2.6 (Gromov-Hausdorff Distance)** Between two compact metric spaces $(X,d_X)$ and $(Y,d_Y)$:
$$d_{GH}(X,Y) = \inf_{f,g,Z} d_H^Z(f(X), g(Y))$$
where the infimum is taken over all metric spaces $Z$ and isometric embeddings $f: X \hookrightarrow Z$ and $g: Y \hookrightarrow Z$. Independent of any specific coordinate system, this characterizes the intrinsic distance between the "shapes" of two manifolds, and is used in RFP to describe the convergence of the agent manifold toward the true knowledge manifold.
