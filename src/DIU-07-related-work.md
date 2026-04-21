# DIU §13：相关工作

> 本章将 DIU 框架置于现有理论谱系中，厘清继承、扩展与根本差异。

---

## §13.1 信息几何（Information Geometry）

### 核心成果回顾

Amari（1985, 2016）[R7, R8] 将统计流形赋予 Fisher 信息度量，建立了参数化概率模型空间上的黎曼几何。核心结构：

- **统计流形**：参数化分布族 $\{p_\theta\}_{\theta \in \Theta}$ 构成可微流形；
- **Fisher 度量**：$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\!\left[\partial_i \log p_\theta \cdot \partial_j \log p_\theta\right]$；
- **自然梯度**：$\tilde{\nabla}_\theta \mathcal{L} = G(\theta)^{-1} \nabla_\theta \mathcal{L}$，沿统计流形最速下降方向更新。

该框架已被用于分析神经网络训练的曲率（Martens, 2014 [R30]；Pascanu & Bengio, 2013 [R31]）。

### DIU 与信息几何的关系

**继承**：DIU 的知识流形 $\mathcal{M}_K$ 在概念上类比于统计流形——两者都是将"能力结构"赋予几何意义。Wasserstein 距离 $W_2(\mu_A, \mu_B)$ 与 Fisher 度量在特定极限下存在形式对应（Otto 微积分，2001 [R9]）。

**根本差异**：

| 维度 | 信息几何 | DIU |
|---|---|---|
| 几何对象 | 参数空间 $\Theta$ 上的流形 | 知识内容空间 $\mathcal{M}_K$ 上的测度 |
| 核心度量 | Fisher 信息（局部曲率） | Hausdorff 维数 $\beta$（全局覆盖广度） |
| 主要工具 | 黎曼度量、测地线 | 测度论、Radon-Nikodym 导数 |
| 系统比较 | 同一模型族内（$\Theta$ 固定） | 跨架构、跨训练目标 |

**关键补充**：信息几何关注参数空间的局部曲率，回答"如何高效更新 $\theta$"；DIU 关注输出测度在知识流形上的覆盖结构，回答"系统能覆盖哪些知识区域"。两者正交互补——信息几何是训练优化框架，DIU 是能力评估框架。

**命题 13.1（Fisher 度量与 $W_2$ 的对应）**

在局部坐标下，当 $\mu_\theta$ 为高斯族时，$W_2^2(\mu_\theta, \mu_{\theta+d\theta}) = d\theta^\top G(\theta)\, d\theta + O(\|d\theta\|^3)$，即 Wasserstein 距离的黎曼近似恢复为 Fisher 度量。DIU 用全局 $W_2$ 取代局部 Fisher 近似，捕获跨分布的大尺度覆盖差异。

---

## §13.2 拓扑数据分析（Topological Data Analysis, TDA）

### 核心成果回顾

Carlsson（2009）[R10]、Edelsbrunner & Harer（2010）[R4] 建立了数据驱动的拓扑方法体系：

- **持续同调（Persistent Homology）**：对滤流 $\emptyset \subseteq X_0 \subseteq X_1 \subseteq \cdots$ 追踪拓扑特征（连通分量、洞、空腔）的出生/消亡，生成持续图 $\mathrm{PD}_k$；
- **Betti 数**：$\beta_k = \dim H_k(X; \mathbb{F})$，刻画第 $k$ 阶拓扑不变量；
- **Mapper 算法**：在高维数据上可视化拓扑结构；
- **神经网络拓扑分析**：Bianchini & Scarselli（2014）[R32] 用 Betti 数分析网络表达力。

### DIU 与 TDA 的关系

**继承**：DIU 的覆盖广度 $\beta(f) = \dim_H(\operatorname{supp}(\mu_f))$ 与 TDA 的 Betti 数共享同一直觉——测量支撑集的"复杂性"或"维度"。WIP-5 的方案 A（持续同调梯度估计）直接复用 TDA 工具作为 $\hat{\beta}$ 的可微代理。

**根本差异**：

| 维度 | TDA | DIU |
|---|---|---|
| 分析对象 | 数据点云的拓扑 | 知识测度 $\mu_f$ 的支撑结构 |
| 核心量 | Betti 数（离散整数） | Hausdorff 维数（连续实数） |
| 主要应用 | 数据可视化、特征提取 | 智能能力的统一定量框架 |
| 参数依赖 | 纯数据驱动 | 依赖 RFP（嵌入忠实公设） |

**注记 13.1**：Hausdorff 维数 $\dim_H$ 与持续同调的"持续性积分"$\sum_{(b,d)} (d-b)^p$ 在某些条件下可以互相近似（Schweinhart, 2020 [R12]），这为 WIP-5 方案 A 的保真度提供了理论依据，但一般情形下两者刻画的是不同的几何特征（维数 vs 拓扑不变量）。

---

## §13.3 最优传输理论（Optimal Transport）

### 核心成果回顾

Villani（2003, 2009）[R2]、Santambrogio（2015）[R3] 系统建立了最优传输理论：

- **Wasserstein 距离**：$W_p(\mu, \nu) = \left(\inf_{\gamma \in \Pi(\mu,\nu)} \int d(x,y)^p d\gamma(x,y)\right)^{1/p}$；
- **Brenier 定理**：凸代价下最优传输映射存在且唯一（绝对连续测度之间）；
- **Otto 微积分**：$W_2$ 空间上的黎曼结构，将偏微分方程视为测度空间中的梯度流；
- **神经网络应用**：生成模型（Arjovsky et al., 2017 [R26]，WGAN），分布对齐，领域自适应。

### DIU 与最优传输的关系

**继承**：DIU 的模型间结构距离 $\Delta(f_A, f_B) = W_2(\mu_A, \mu_B)$（§10.2）直接采用 Wasserstein 距离作为核心度量，继承了其几何意义：$W_2$ 不仅测量分布差异，而且编码了**如何将一个知识覆盖结构"传输"为另一个**的最小代价，比 KL 散度更符合知识结构比较的直觉。

**扩展**：DIU 在最优传输工具之上加入了 Hausdorff 维数分析和层级分类，这是最优传输理论本身未涉及的能力评估维度。

**命题 13.2（$W_2$ 的几何信息优势）**

设 $f_A, f_B \in \mathcal{D}_1$，$\mu_A, \mu_B$ 具有相同均值和方差。则：
$$D_{KL}(\mu_A \| \mu_B) \to 0 \not\Rightarrow W_2(\mu_A, \mu_B) \to 0,$$
反之亦然（KL 散度对支撑差异不敏感，$W_2$ 对几何位移敏感）。因此 $\Delta(f_A, f_B) = W_2$ 能检测 KL 散度无法检测的覆盖结构差异——例如两个模型的 benchmark 得分相同，但知识分布在流形上的几何位置完全不同。

---

## §13.4 Scaling Laws 与幂律理论

### 核心成果回顾

Kaplan et al.（2020）[R17]、Hoffmann et al.（2022，Chinchilla）[R18] 建立了大语言模型的经验规律：

- **幂律关系**：$\mathcal{L}(N) \approx \left(\frac{N_0}{N}\right)^\alpha$（损失随参数量的幂律下降）；
- **最优计算分配**：给定训练计算预算 $C$，最优参数量 $N^* \propto C^{0.5}$（Chinchilla 定律）；
- **数据墙**：随着数据需求增长，互联网规模的高质量文本数据趋于耗尽（Villalobos et al., 2022 [R33]）。

### DIU 与 Scaling Laws 的关系

**继承**：WIP-2 中 $\beta(f) = \eta(f) \cdot \mathcal{B}(f)$ 的乘法结构与 Scaling Laws 精神相符：算力（测度预算 $\mathcal{B}$）对能力（覆盖广度 $\beta$）有直接贡献。

**根本差异与扩展**：

**命题 13.3（Scaling Laws 的 DIU 解释与局限）**

Scaling Law 的幂律关系 $\mathcal{L}(N) \propto N^{-\alpha}$ 等价于：随参数量增加，系统在**训练分布支撑附近**的局部密度 $\rho_{f_\theta}$ 以幂律速率提升。然而：

1. $\mathcal{L}(N)$ 是 CE 损失的代理——由命题 WIP-5.1，CE 只测量局部密度，不测量覆盖广度 $\beta$；
2. Scaling Law 描述 $\mathcal{D}_1$ **内部**的连续改进，无法刻画跨层级跃迁；
3. 推论 8.1（§8）已严格证明：Scaling Laws 无论外推多远，均无法跨越 $\mathcal{D}_1 \to \mathcal{D}_2$ 的基数壁垒。

**注记 13.2（数据墙的测度论解释）**

互联网文本数据的"耗尽"在 DIU 框架下有精确刻画：训练数据 $\mathcal{D}$ 的支撑是 $\mathcal{M}_K$ 上的有限子集，其所能提供的 $\beta$ 上界由 $\dim_H(\operatorname{supp}(\mathcal{D}))$ 决定。当数据规模饱和后，继续增加同质数据无法提升 $\beta$ 的上界——这是数据墙的几何本质，不仅是经验现象。

---

## §13.5 VC 维与 PAC 学习理论

### 核心成果回顾

Vapnik & Chervonenkis（1971）[R15]、Valiant（1984）[R34] 建立了学习的统计理论：

- **VC 维**：假设类 $\mathcal{H}$ 的 VC 维 $d = \mathrm{VCdim}(\mathcal{H})$ 刻画其打散点集的最大基数；
- **PAC 学习**：以样本复杂度 $m = O(d \cdot \varepsilon^{-2} \log(1/\delta))$ 保证泛化误差；
- **Rademacher 复杂度**：更细粒度的容量测量，与 VC 维有阶关系；
- **深度网络的 VC 维**：Bartlett et al.（2019）[R14] 给出 $O(WL\log W)$ 的上界（$W$ 参数数，$L$ 层数）。

### DIU 与 PAC 学习的关系

**继承**：VC 维是容量（capacity）的组合测量；DIU 的覆盖广度 $\beta$ 是容量的测度论测量。两者共享同一直觉：更高的"容量"意味着能表示更多的函数类。

**根本差异**：

| 维度 | PAC / VC 理论 | DIU |
|---|---|---|
| 容量测量 | 离散（可打散的点集基数） | 连续（Hausdorff 维数） |
| 分析对象 | 假设类 $\mathcal{H}$（静态） | 系统测度 $\mu_f$（连续） |
| 理论目标 | 泛化误差上界 | 能力覆盖结构（不限于泛化） |
| 层级观 | 单层容量 | 跨层级（$\mathcal{D}_0\sim\mathcal{D}_\infty$）质性区别 |

**命题 13.4（VC 维与 $\beta$ 的关系）**

设假设类 $\mathcal{H}$ 的 VC 维为 $d$，对应测度 $\mu_\mathcal{H}$ 的覆盖广度为 $\beta(\mathcal{H})$。则：
$$\beta(\mathcal{H}) \leq d \cdot \log 2 + O(1),$$

其中等号在 $\mathcal{H}$ 的 VC 维由嵌入空间的 Hausdorff 维数直接决定时（如线性分类器类）成立。一般情形下，$\beta$ 提供比 VC 维更细粒度的测量（例如 MoE 架构的 $\beta$ 在技术上高于同参数量 Dense 网络，但两者具有可比的 VC 维上界）。

**注记 13.3（PAC 框架的覆盖盲点）**

PAC 理论的泛化误差分析依赖于训练分布与测试分布**同源**的假设（i.i.d.）。当测试分布与训练分布存在覆盖结构差异（即测试查询落入 $\mathcal{V}(f,\varepsilon,\tau)$）时，PAC 上界失效——蝴蝶效应定理（§9）正刻画了这种失效的测度论机制。DIU 的脆弱图谱 $\mathcal{V}$ 可视为对 PAC 框架的"分布偏移脆弱性"补充。

---

## §13.6 综合定位

**表 13.1：相关理论框架与 DIU 的系统对比**

| 理论框架 | DIU 继承 | DIU 扩展 | DIU 独有 |
|---|---|---|---|
| 信息几何 | $W_2$ ≈ Fisher 度量（局部） | 全局覆盖结构分析 | 稠密度层级、基数天花板 |
| TDA | 持续同调工具（WIP-5 方案A） | 覆盖广度作为核心指标 | RFP 公设、$\mathcal{D}_0\sim\mathcal{D}_\infty$ 框架 |
| 最优传输 | $W_2$ 作为核心距离 | 测度覆盖广度 $\beta$ | 层级跃迁的测度论证明 |
| Scaling Laws | 算力-能力乘法结构 | $\beta$ 的硬上界分析 | 层级壁垒不可逾越性证明 |
| PAC / VC 理论 | 容量-泛化关系直觉 | $\beta$ 比 VC 维更细粒度 | 脆弱图谱、分布偏移失效机制 |

**核心创新定位**：上述所有框架均在不同角度刻画了"智能的某一侧面"，但没有一个框架能同时回答以下三个问题：

1. **统一性**：不同架构（Transformer/Mamba/扩散模型）是否有统一的能力度量？（定理 7.1 + §10）
2. **天花板**：当前所有 LLM 与 AGI 之间是否存在原则性边界？（定理 8.1 + WIP-4）
3. **实践**：如何从覆盖结构出发改进训练目标和评估体系？（WIP-5 + 命题 10.1）

DIU 的贡献是在同一测度论框架下，同时给出三个问题的定量化回答。
