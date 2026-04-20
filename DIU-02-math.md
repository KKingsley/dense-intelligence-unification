# DIU §2：数学预备

---

## §2 数学预备

### 2.1 测度空间

**定义 2.1（测度空间）** 三元组 $(\Omega, \Sigma, \mu)$ 称为测度空间，其中 $\Omega$ 为样本空间，$\Sigma$ 为 $\sigma$-代数，$\mu: \Sigma \to [0,+\infty]$ 满足可数可加性。

**定义 2.2（绝对连续性）** 设 $\mu, \lambda$ 为 $(\Omega, \Sigma)$ 上的两个测度。若对任意 $A \in \Sigma$，$\lambda(A)=0 \Rightarrow \mu(A)=0$，则称 $\mu$ 关于 $\lambda$ **绝对连续**，记为 $\mu \ll \lambda$。

**定理 2.1（Radon-Nikodym）** 若 $\mu \ll \lambda$ 且 $\lambda$ 为 $\sigma$-有限测度，则存在唯一（$\lambda$-a.e.）非负可测函数 $\rho = d\mu/d\lambda$，使得：
$$\mu(A) = \int_A \rho \, d\lambda, \quad \forall A \in \Sigma$$
$\rho$ 称为**密度函数**，是 DIU 中"局部稠密程度"的形式化对象。

### 2.2 Hausdorff 维数

**定义 2.3（Hausdorff 测度）** 对度量空间 $(X,d)$ 中的集合 $E$，$s$ 维 Hausdorff 测度：
$$\mathcal{H}^s(E) = \lim_{\delta \to 0} \inf\left\{\sum_i \operatorname{diam}(U_i)^s : E \subseteq \bigcup_i U_i,\; \operatorname{diam}(U_i) < \delta\right\}$$

**定义 2.4（Hausdorff 维数）**
$$\dim_H(E) = \inf\{s \geq 0 : \mathcal{H}^s(E) = 0\} = \sup\{s \geq 0 : \mathcal{H}^s(E) = \infty\}$$
刻画覆盖集合的"分形广度"——DIU 中用于衡量智能覆盖的结构性宽度。

### 2.3 Wasserstein 距离

**定义 2.5（$p$-Wasserstein 距离）** 设 $\mu, \nu$ 为度量空间 $(X,d)$ 上的概率测度，$\Gamma(\mu,\nu)$ 为边缘分布分别为 $\mu,\nu$ 的所有联合分布之集合：
$$W_p(\mu,\nu) = \left(\inf_{\gamma \in \Gamma(\mu,\nu)} \int_{X \times X} d(x,y)^p \, d\gamma(x,y)\right)^{1/p}$$

**注记 2.1** $W_2$ 距离的几何意义：将概率质量从 $\mu$ 搬运至 $\nu$ 的最小代价（以欧氏距离平方计量）。$W_2$ 赋予概率测度空间黎曼流形结构（Otto 微积分），DIU 中用于比较两模型覆盖结构的整体差异。

### 2.4 Gromov-Hausdorff 距离

**定义 2.6（Gromov-Hausdorff 距离）** 两紧致度量空间 $(X,d_X)$，$(Y,d_Y)$ 之间：
$$d_{GH}(X,Y) = \inf_{f,g,Z} d_H^Z(f(X), g(Y))$$
下确界取遍所有度量空间 $Z$ 及等距嵌入 $f: X \hookrightarrow Z$，$g: Y \hookrightarrow Z$。与具体坐标系无关，刻画两流形"形状"的本质距离，用于 RFP 中描述代理流形向真实知识流形的收敛。
