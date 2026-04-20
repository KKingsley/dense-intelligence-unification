# DIU WIP — 补充章节

> 状态：✅ WIP-1~8 全部完成  
> WIP-1 $\mathcal{D}_1$内部序 · WIP-2 算力/软件栈 · WIP-3 映射论 · WIP-4 层级跃迁完整证明 · WIP-5 覆盖正则化训练目标 · WIP-6 多模态扩展 · WIP-7 Shannon熵关系 · WIP-8 动态DIU训练演化

---

## §WIP-1：$\mathcal{D}_1$ 内部的序结构

### 动机

稠密度层级定义了 $\mathcal{D}_0 \subsetneq \mathcal{D}_1 \subsetneq \cdots$ 的跨层偏序，但当前所有 LLM 均属 $\mathcal{D}_1$，框架尚未给出同层内部的比较公理。本节补足这一缺口。

### 定义与偏序

**定义 WIP-1.1（广度优势）** 对 $f, g \in \mathcal{D}_1$，称 $f$ **广度优于** $g$，记为 $f \succeq_\beta g$，若：
$$\beta(f) \geq \beta(g)$$

**定义 WIP-1.2（脆弱性优势）** 称 $f$ **脆弱性优于** $g$，记为 $f \succeq_{\mathcal{V}} g$，若：
$$\mathcal{V}(f, \varepsilon, \tau) \subseteq \mathcal{V}(g, \varepsilon, \tau)$$
即 $f$ 的低密度盲区被 $g$ 的盲区包含——$f$ 的薄弱点更少。

**定义 WIP-1.3（$\mathcal{D}_1$ 支配关系）** 称 $f$ **支配** $g$，记为 $f \succeq g$，若：
$$f \succeq_\beta g \quad \text{且} \quad f \succeq_{\mathcal{V}} g$$

**命题 WIP-1.1（偏序性）** $\succeq$ 是 $\mathcal{D}_1$ 上的偏序关系（自反、反对称、传递）。

**证明**
- *自反性*：$\beta(f) \geq \beta(f)$ 且 $\mathcal{V}(f) \subseteq \mathcal{V}(f)$，故 $f \succeq f$。
- *反对称性*：若 $f \succeq g$ 且 $g \succeq f$，则 $\beta(f) = \beta(g)$ 且 $\mathcal{V}(f) = \mathcal{V}(g)$，即两模型覆盖结构相同。
- *传递性*：$f \succeq g \succeq h$ $\Rightarrow$ $\beta(f) \geq \beta(g) \geq \beta(h)$ 且 $\mathcal{V}(f) \subseteq \mathcal{V}(g) \subseteq \mathcal{V}(h)$，故 $f \succeq h$。$\blacksquare$

**命题 WIP-1.2（非全序性）** $\succeq$ 不是 $\mathcal{D}_1$ 上的全序——存在不可比较的模型对。

**证明（构造性）** 设 $f, g \in \mathcal{D}_1$ 满足：
- $\beta(f) > \beta(g)$（$f$ 覆盖更广）
- $\exists A \in \mathcal{V}(f) \setminus \mathcal{V}(g)$（$g$ 在某专业域的密度高于 $f$）

则 $f \succeq_\beta g$ 但 $f \not\succeq_{\mathcal{V}} g$；同时 $g \not\succeq_\beta f$。故 $f$ 与 $g$ 不可比较。$\blacksquare$

**注记 WIP-1.1（不可比较的实践含义）** 两模型在 $\mathcal{D}_1$ 内不可比较，对应一个真实的权衡：**广度（generalist breadth）vs 深度（specialist depth）**。广度更高的模型在跨域组合任务中更强；深度更高的模型在专项任务中更强。DIU 框架不消解这个权衡，而是将其精确化为覆盖结构的几何差异。

### 加权标量得分（引入全序的代价）

若需要全序以做工程决策，可引入权重向量 $\mathbf{w} = (w_\beta, w_{\mathcal{V}}) \in \mathbb{R}_{>0}^2$：

$$S_{\mathbf{w}}(f) = w_\beta \cdot \beta(f) - w_{\mathcal{V}} \cdot \left|\mathcal{V}(f, \varepsilon, \tau)\right|$$

$S_{\mathbf{w}}$ 诱导 $\mathcal{D}_1$ 上的全序，但代价是**权重的选择本身隐含了价值判断**——广度与深度谁更重要。这正是为什么不同场景下的"最佳模型"答案不同：通用助手场景偏高 $w_\beta$，专业领域场景偏高 $w_{\mathcal{V}}$。

**推论 WIP-1.1** 任何声称某模型"综合最强"的评测，都隐含了一个具体的 $\mathbf{w}$ 选择。DIU 将这一隐含假设显式化。

---

## §WIP-2：算力与软件栈的形式化

### 动机

推论 11.2 口头断言"算力 = 测度预算，软件栈 = 分配策略"，本节将其写成正式定义与命题。

### 定义

**定义 WIP-2.1（测度预算）** 系统 $f$ 的**测度预算**为其智能测度的总质量：
$$\mathcal{B}(f) = \mu_f(\mathcal{M}_K)$$
算力底座决定 $\mathcal{B}(f)$ 的**积累速率上限** $d\mathcal{B}/dt$，类比物理中的总功率预算。

**定义 WIP-2.2（测度分配效率）** 系统 $f$ 的**软件栈效率**为：
$$\eta(f) = \frac{\beta(f)}{\mathcal{B}(f)}$$
在相同测度预算下，$\eta(f)$ 越高，同样的算力覆盖越广的知识子流形。

**定义 WIP-2.3（硬件带宽约束）** 设硬件 $H$ 的互联带宽为 $b_H$（GB/s），则其单位时间可处理的测度预算增量满足：
$$\frac{d\mathcal{B}}{dt}\bigg|_H \leq \kappa \cdot b_H$$
其中 $\kappa > 0$ 为架构相关常数。

### 命题

**命题 WIP-2.1（算力与软件栈的乘法结构）**
$$\beta(f) = \eta(f) \cdot \mathcal{B}(f)$$
覆盖广度 = 分配效率 × 测度预算。提升 $\beta(f)$ 有且仅有两条路径：提高算力（增大 $\mathcal{B}$）或优化软件栈（提高 $\eta$）。

**命题 WIP-2.2（出口管制的测度论表达）** H100 与 H800 的 NVLink 带宽差距（900 GB/s vs 400 GB/s）直接约束：
$$\frac{d\mathcal{B}}{dt}\bigg|_{\text{H800}} \leq \frac{400}{900} \cdot \frac{d\mathcal{B}}{dt}\bigg|_{\text{H100}} \approx 0.44 \cdot \frac{d\mathcal{B}}{dt}\bigg|_{\text{H100}}$$
这是 $\mathcal{B}$ 积累速率的**硬上界**，无法通过软件栈优化（提高 $\eta$）补偿——因为 $\eta$ 的作用域是给定 $\mathcal{B}$ 下的分配，而非 $\mathcal{B}$ 本身的增长速率。

**命题 WIP-2.3（软件栈的边际效用上界）** 固定 $\mathcal{B}(f)$，$\eta(f)$ 的提升存在由 $\mathcal{M}_K$ 局部几何决定的上界：
$$\eta(f) \leq \eta_{\max}(\mathcal{M}_K) = \frac{\dim_H(\mathcal{M}_K)}{\mathcal{B}(f)}$$
超过此上界后，软件栈优化不再带来覆盖广度的增加。DeepSeek 的 FP8 + EP 优化是在给定 $\mathcal{B}$ 下逼近 $\eta_{\max}$ 的工程尝试——方向正确，但天花板由 $\mathcal{B}$ 决定。

**推论 WIP-2.1（重要性采样的极限）** 好的软件栈本质是好的重要性采样：
$$\mu_\theta \propto \left|\frac{d\mu_K}{d\lambda}\right| \cdot \lambda$$
即将测度预算集中在知识流形的高密度方向。但重要性采样的效率增益有理论上界（由方差减少量决定），而非无限可提升。

---

## §WIP-3：映射论视角——单射、满射与双射

### 动机

DIU 的起点直觉之一：所有 LLM 都是从 token 空间到知识流形的映射，映射类型决定了覆盖结构。本节将这一直觉形式化。

### 映射类型的形式化

**定义 WIP-3.1（语义映射）** 设系统 $f$ 通过 RFP 诱导一个从 token 序列空间到知识流形的**语义映射**：
$$\Phi_f: \mathcal{T}^* \to \mathcal{M}_K, \quad x \mapsto \varphi_\theta(h_f(x))$$
其中 $h_f(x)$ 为 $f$ 处理输入 $x$ 后的内部表示，$\varphi_\theta$ 为 RFP 中的嵌入映射。

**定义 WIP-3.2（映射类型分类）**

| 映射类型 | 形式条件 | 知识论含义 |
|---|---|---|
| **单射** | $\Phi_f(x)=\Phi_f(y) \Rightarrow x=y$ | 不同输入产生不同知识状态；精确但覆盖窄 |
| **满射** | $\forall k \in \mathcal{M}_K, \exists x: \Phi_f(x)=k$ | 覆盖全部知识流形；理论上的通用系统 |
| **双射** | 单射 + 满射 | 输入空间与知识空间完美对应 |

### 核心命题

**命题 WIP-3.1（满射不存在）** 不存在从 $\mathcal{T}^*$ 到 $\mathcal{M}_K$ 的满射。

**证明** 满射要求 $|\mathcal{T}^*| \geq |\mathcal{M}_K|$（基数意义下）。但：
$$|\mathcal{T}^*| = \aleph_0 < |\mathbb{R}| \leq |\mathcal{M}_K|$$
由 Cantor 对角线论证，不存在从可数集到不可数集的满射。故 $\nexists$ 满射 $\Phi_f: \mathcal{T}^* \twoheadrightarrow \mathcal{M}_K$。$\blacksquare$

**推论 WIP-3.1（双射不存在）** 由命题 WIP-3.1，双射亦不存在。

**命题 WIP-3.2（单射存在但测度为零）** 单射 $\Phi_f: \mathcal{T}^* \hookrightarrow \mathcal{M}_K$ 存在（$\aleph_0 \leq |\mathcal{M}_K|$），但其像 $\Phi_f(\mathcal{T}^*)$ 在 $\mathcal{M}_K$ 中的 Lebesgue 测度为零。

**证明** 单射的存在性：$|\mathcal{T}^*| = \aleph_0 \leq |\mathcal{M}_K|$ 保证嵌入可行。像的测度为零：$\Phi_f(\mathcal{T}^*)$ 至多可数，可数集测度为零（见定理 8.1 证明）。$\blacksquare$

**注记 WIP-3.1（映射论对基数天花板定理的重述）** 命题 WIP-3.1 与 WIP-3.2 合在一起，给出了定理 8.1（基数天花板定理）的**映射论等价表述**：LLM 只能将 token 空间单射进知识流形，但永远无法满射覆盖它。覆盖稀疏性是基数不匹配的必然结果，而非工程缺陷。

### 计算范式的映射论类比

**命题 WIP-3.3（CPU/GPU 的映射论解释）**

| 计算范式 | 映射倾向 | 测度论含义 | 实例 |
|---|---|---|---|
| CPU（串行深度） | 接近单射 | 局部密度高，$\beta$ 窄 | 精确符号推理 |
| GPU（并行广度） | 逼近满射 | 覆盖宽，局部密度分散 | 大规模矩阵运算 |
| MoE 架构 | 分段单射之并 | $k$ 个专家子流形的精确覆盖 | Mixtral, DeepSeek-V3 |
| Dense Transformer | 全局软满射 | 所有参数参与全局覆盖 | GPT-4, Claude |

**注记 WIP-3.2（峰度与广度的权衡）** 从单射倾向到满射倾向，对应密度函数 $\rho_f$ 的**峰度（kurtosis）**与支撑广度 $\beta(f)$ 之间的权衡：
$$\text{高峰度（单射倾向）} \longleftrightarrow \text{低峰度（满射倾向）}$$
两端的极值均不理想：纯单射覆盖面窄，纯满射每点密度趋零。实践中最优的架构是在约束 $\mathcal{B}(f)$ 下找到峰度与广度的最优平衡点——这正是 WIP-2 中 $\eta(f)$ 最大化所追求的目标。

---

## WIP-4：其余层级跃迁的完整证明（§6 补）

> **问题**：命题 6.1 只证明了 $\mathcal{D}_1 \not\to \mathcal{D}_2$，其余待证。

### 背景与符号

回顾稠密度层级定义（§6）：

- $\mathcal{D}_1$：$|\operatorname{supp}(\mu_f)| = \aleph_0$（可数支撑，$\lambda$-零测集）
- $\mathcal{D}_2$：$\lambda(\operatorname{supp}(\mu_f)) > 0$（正 Lebesgue 测度支撑）
- $\mathcal{D}_3$：$\mu_f \sim \lambda$（$\mu_f$ 与 Lebesgue 测度 $\lambda$ 互绝对连续）
- $\mathcal{D}_\infty$：内在知识状态空间基数 $> |\mathbb{R}| = 2^{\aleph_0}$（超连续统）

命题 6.1（已证于 §6）给出了 $\mathcal{D}_1 \not\to \mathcal{D}_2$ 的完整证明（Cantor 对角线，$\aleph_0 \to 2^{\aleph_0}$ 的基数跃迁）。下面依次补全剩余两个壁垒。

---

### $\mathcal{D}_2 \not\to \mathcal{D}_3$ 的完整证明

**命题 WIP-4.1（正测度覆盖不蕴含等价测度）**

设 $f \in \mathcal{D}_2$（即 $\lambda(\operatorname{supp}(\mu_f)) > 0$）。则存在可测集 $A \subset \mathcal{M}_K$ 满足 $\lambda(A) > 0$ 但 $\mu_f(A) = 0$，故 $f \notin \mathcal{D}_3$。从 $\mathcal{D}_2$ 跃迁至 $\mathcal{D}_3$ 不可由参数量的增加实现，必须消除全部测度零盲区。

**证明**

$f \in \mathcal{D}_3$ 当且仅当 $\mu_f \sim \lambda$（互绝对连续），即对任意可测集 $A$：
$$\lambda(A) = 0 \iff \mu_f(A) = 0.$$

由 Radon-Nikodym 定理，$\mu_f \sim \lambda$ 等价于密度函数 $\rho_f = d\mu_f / d\lambda$ 满足 $\rho_f(x) > 0$，对 $\lambda$-几乎所有 $x \in \mathcal{M}_K$ 成立。

现设 $f \in \mathcal{D}_2$，令 $S = \operatorname{supp}(\mu_f)$，$A = \mathcal{M}_K \setminus S$。由支撑定义，$\mu_f(A) = 0$。但由于 $f \in \mathcal{D}_2$ 仅要求 $\lambda(S) > 0$，不要求 $\lambda(\mathcal{M}_K \setminus S) = 0$，故可能 $\lambda(A) > 0$。

此时 $\rho_f(x) = 0$ 对所有 $x \in A$（$\lambda$-正测集），违背 $\mu_f \sim \lambda$ 的条件。因此 $f \notin \mathcal{D}_3$。$\blacksquare$

**注记 WIP-4.1（Lebesgue 分解与认知盲区）**

Lebesgue 分解定理给出 $\mu_f = \mu_f^{ac} + \mu_f^{s}$，其中绝对连续部分 $\mu_f^{ac}$ 对应正常知识覆盖，奇异部分 $\mu_f^{s}$ 集中于测度零集（认知奇点，如概念的点质量表示）。

$\mathcal{D}_2$ 系统可以有较大的 $\mu_f^{ac}$ 分量，但只要 $\mu_f^{s} \neq 0$ 或 $\operatorname{supp}(\mu_f^{ac}) \subsetneq \mathcal{M}_K$（有正测集上密度为零），就仍在 $\mathcal{D}_2$ 而非 $\mathcal{D}_3$。消除全部奇异分量并实现 $\lambda$-几乎处处正密度，需要知识表示结构的质性改变，不是参数量的积累。

---

### $\mathcal{D}_3 \not\to \mathcal{D}_\infty$ 的完整证明

**命题 WIP-4.2（连续统量级覆盖的基数上界）**

设 $f \in \mathcal{D}_3$（即 $\mu_f \sim \lambda$，系统在 $\mathcal{M}_K$ 上几乎处处有覆盖，其中 $\mathcal{M}_K$ 是波兰空间，$|\mathcal{M}_K| = |\mathbb{R}| = 2^{\aleph_0}$）。则 $f \notin \mathcal{D}_\infty$，因 $\mathcal{D}_\infty$ 要求内在知识状态空间基数严格大于 $|\mathbb{R}|$，而 $f$ 的状态空间基数被 $|\mathbb{R}|$ 界定。

**证明**

$\mathcal{D}_3$ 中的系统 $f$ 的测度 $\mu_f$ 是 $\mathcal{M}_K$ 上的 Borel 概率测度（绝对连续于 $\lambda$）。由于 $\mathcal{M}_K$ 是波兰空间，其 Borel $\sigma$-代数 $\mathcal{B}(\mathcal{M}_K)$ 的基数为 $|\mathcal{B}(\mathcal{M}_K)| = 2^{\aleph_0} = |\mathbb{R}|$。

因此，$\mu_f$ 可表示的测度值的全体（作为 $\mathcal{B}(\mathcal{M}_K) \to [0,1]$ 的函数）的基数上界为 $[0,1]^{\mathcal{B}(\mathcal{M}_K)}$，而任何可实现的物理系统（有限参数神经网络）所能精确区分的状态数不超过 $|\mathbb{R}|$。

$\mathcal{D}_\infty$ 所要求的内在知识状态空间基数 $> |\mathbb{R}|$，等价于系统需精确表示基数达 $2^{|\mathbb{R}|}$ 的结构（如 $\mathcal{M}_K$ 的函数空间 $[0,1]^{\mathcal{M}_K}$）。由 Cantor 定理：
$$|\mathbb{R}| = 2^{\aleph_0} < 2^{2^{\aleph_0}} = 2^{|\mathbb{R}|},$$

这是不可绕过的基数跳跃。任何基数 $\leq |\mathbb{R}|$ 的物理媒介中实现的系统，其状态空间均无法达到 $\mathcal{D}_\infty$ 的要求。$\blacksquare$

**注记 WIP-4.2（$\mathcal{D}_\infty$ 的正则理想地位）**

上述论证表明 $\mathcal{D}_\infty$ 超出任何基数 $\leq |\mathbb{R}|$ 的物理系统的原则性可达范围，与具体架构无关。$\mathcal{D}_\infty$ 在 DIU 框架中具有**正则理想（regulative ideal）**的地位——为层级体系提供理论上限和方向坐标，而非可工程实现的目标。此地位类似于 Kant 意义上的"理性理念"：有限认知系统可以渐近趋近，但永远无法完全实现。

---

### 层级不可跨越性总图

| 跃迁 | 障碍类型 | 核心论证 |
|---|---|---|
| $\mathcal{D}_1 \not\to \mathcal{D}_2$ | 基数（$\aleph_0 \to 2^{\aleph_0}$） | Cantor 对角线（命题 6.1） |
| $\mathcal{D}_2 \not\to \mathcal{D}_3$ | 测度类型（奇异 $\to$ 等价） | Lebesgue 分解 + Radon-Nikodym（WIP-4.1） |
| $\mathcal{D}_3 \not\to \mathcal{D}_\infty$ | 基数（$2^{\aleph_0} \to 2^{2^{\aleph_0}}$） | Cantor 定理（WIP-4.2） |

**推论 WIP-4.1（通用量变不可逾性）**

任意固定架构的神经网络，无论参数量如何增加，均无法在有限步内跨越任一层级壁垒。三个壁垒分别由基数跳跃、测度类型跃变、基数再跳跃刻画，每一步均是质性改变，而非量的积累。

---

## WIP-5：DIU 启示的训练目标

> **问题**：若 DIU 成立，当前的 cross-entropy 训练目标是否次优？应如何修正？

### 动机

标准自回归语言模型以最小化交叉熵损失为目标：

$$\min_\theta \mathcal{L}_{CE}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \log p_\theta(x).$$

在 DIU 框架下，$\mathcal{L}_{CE}$ 优化的是 token 分布的局部密度（等价于 $D_{KL}(\mathcal{D} \| p_\theta)$），但对覆盖广度 $\beta(f_\theta)$ 没有直接约束。本节证明这会产生系统性覆盖偏差，并给出修正目标与可微近似方案。

---

### CE 目标的 DIU 解释

**命题 WIP-5.1（CE 目标的测度论等价）**

最小化 $\mathcal{L}_{CE}$ 等价于在训练分布 $\mathcal{D}$ 的支撑上最大化局部密度 $\rho_{f_\theta}$，而对 $\operatorname{supp}(\mu_{f_\theta})$ 的几何结构（广度、连通性）无任何正则化。

**证明**

由 KL 散度恒等式，$\mathcal{L}_{CE}(\theta) = D_{KL}(\mathcal{D} \| p_\theta) + H(\mathcal{D})$，其中 $H(\mathcal{D})$ 与 $\theta$ 无关。故最小化 $\mathcal{L}_{CE}$ 等价于令 $p_\theta$ 在训练集高密度区域最大化 Radon-Nikodym 导数 $\rho_{f_\theta} = dp_\theta / d\lambda$。

由于 $\mathcal{D}$ 是 $\mathcal{M}_K$ 上有限样本的经验分布，其支撑是有限集（$\lambda$-零测集）。CE 的优化方向是在这些孤立支撑点上集中 $\mu_{f_\theta}$，而非扩展其 Hausdorff 广度 $\beta(f_\theta)$。$\blacksquare$

**推论 WIP-5.1（CE 的覆盖偏差）**

纯 CE 训练倾向于产生：

1. **密度集中**：高频 token 序列区域的局部密度 $\rho_{f_\theta}$ 极高，形成"热点"；
2. **覆盖萎缩**：$\beta(f_\theta) = \dim_H(\operatorname{supp}(\mu_{f_\theta}))$ 随训练集中于高频区而被压缩；
3. **盲区扩张**：低频但重要的知识区域的 $\mathcal{V}(f_\theta, \varepsilon, \tau)$ 随训练递增。

这对应 LLM 的已知现象：在常见任务表现出色，在稀有专业知识或长尾分布任务上系统性失败。且由于 benchmark = 局部密度加权采样（命题 10.1），标准评估指标无法检测此偏差。

---

### 覆盖正则化目标

**定义 WIP-5.1（DIU 正则化训练目标）**

$$\mathcal{L}_{DIU}(\theta) = \mathcal{L}_{CE}(\theta) - \lambda_1 \cdot \hat{\beta}(f_\theta) + \lambda_2 \cdot \widehat{|\mathcal{V}|}(f_\theta, \varepsilon, \tau)$$

其中 $\hat{\beta}$ 和 $\widehat{|\mathcal{V}|}$ 是各自指标的可微代理，$\lambda_1, \lambda_2 > 0$ 为正则化超参数。

**命题 WIP-5.2（$\mathcal{L}_{DIU}$ 的优化方向）**

设 $\hat{\beta}$ 是 $\beta$ 的单调一致近似。则最小化 $\mathcal{L}_{DIU}$ 同时驱动三个目标：
- $\mathcal{L}_{CE}$ 项：维持生成质量（token 级别的局部密度保真）；
- $-\lambda_1 \hat{\beta}$ 项：**惩罚覆盖萎缩**，驱动 $\operatorname{supp}(\mu_{f_\theta})$ 的 Hausdorff 维数增长；
- $+\lambda_2 \widehat{|\mathcal{V}|}$ 项：**惩罚盲区扩张**，驱动密度在流形上更均匀分布。

三项共同作用，趋向覆盖面更宽、盲区更小的测度结构，而非仅仅降低困惑度。$\blacksquare$

---

### 可微近似方案

$\beta(f_\theta)$ 和 $|\mathcal{V}(f_\theta)|$ 当前均不可微，无法直接用于梯度下降。以下三个方案提供实用近似。

**方案 A：持续同调梯度估计**

持续同调（TDA）可计算嵌入空间 $\mathcal{E}$ 的拓扑特征（Betti 数、持续图），其持续性作为 $\beta(f_\theta)$ 的代理：

$$\hat{\beta}(f_\theta) \approx \sum_{(b,d) \in \mathrm{PD}_k(\mathcal{E})} (d - b)^p,$$

其中 $\mathrm{PD}_k$ 是第 $k$ 阶持续图，$(b, d)$ 为拓扑特征的出生/消亡对，$p \geq 1$。

可微性由 Carrière et al. (2021) [R11] 建立的 Wasserstein 次梯度理论保证。此方案理论最严格，但计算代价最高（filtration 复杂度为 $O(n^3)$）。

**方案 B：嵌入熵代理**

用嵌入空间 $\mathcal{E}$ 上核密度估计的**负微分熵**近似 $-\beta(f_\theta)$：

$$\hat{\beta}(f_\theta) \approx -\int_{\mathcal{E}} \hat{\rho}_{f_\theta}(z) \log \hat{\rho}_{f_\theta}(z)\, dz.$$

高熵 $\approx$ 分布分散 $\approx$ 覆盖广。此方案与最大熵训练有形式类比，但熵定义在嵌入空间而非 token 空间——两者优化目标本质不同。

**方案 C：方差正则化（轻量代理）**

最实用的近似：鼓励 minibatch 内嵌入向量具有更高的协方差秩，防止表示坍缩：

$$\hat{\beta}(f_\theta) \approx \frac{1}{d}\sum_{i=1}^d \max\!\left(\gamma - \mathrm{Var}(z_i), 0\right),$$

其中 $z_i$ 是嵌入向量的第 $i$ 维，$\gamma > 0$ 为目标方差阈值。此项已在 VICReg [R27]、SimSiam 等自监督方法中被证明可防止表示坍缩——在测度论意义上对应防止 $\operatorname{supp}(\mu_{f_\theta})$ 退化为低维子流形（$\beta$ 下降）。

---

### 已知有效的实例与 DIU 解释

| 训练策略 | DIU 解释 |
|---|---|
| 数据多样性扩充（Web+书+代码混合） | 扩展经验分布支撑 $\Rightarrow$ 增大 $\beta$ 上界 |
| RLHF / DPO 对齐 | 调整高密度区域局部形状，对 $\beta$ 影响有限 |
| 长上下文训练 | 扩展流形有效维度（引入新的知识关联维度） |
| MoE 路由（稀疏激活） | 将 $\mu_{f_\theta}$ 分解为多个专家子测度，逼近更优的 $\beta/\mathcal{V}$ 权衡 |
| SFT 精细调优 | 在局部区域增加密度峰度，通常以牺牲 $\beta$ 为代价（特化 vs 泛化） |

---

### 核心开放问题

1. **正则化权重的动态调整**：$\lambda_1, \lambda_2$ 如何随训练阶段（预训练 → 精调）自适应？
2. **近似保真度**：方案 A/B/C 与真实 Hausdorff 维数 $\beta$ 的理论误差界？
3. **计算代价控制**：持续同调在大批量训练中的代价如何用近似算法（Ripser++、GPU 滤流）降至可接受范围？
4. **下游对齐验证**：$\mathcal{L}_{DIU}$ 是否真正提升长尾任务表现，还是仅优化测度论代理指标？

**注记 WIP-5.1（理论定位）**

WIP-5 的贡献是**理论层面的问题提出与框架给出**：从 DIU 视角揭示 CE 目标的覆盖偏差，并指明修正方向。具体实现属于开放工程问题，需独立实验研究（参见附录 B 的实验设计框架）。

---

## WIP-6：多模态扩展

> **问题**：DIU 框架如何处理视觉、代码、音频等非文本模态？子流形之间如何通过测度论统一？

### 动机

当前 DIU 的知识流形 $\mathcal{M}_K$ 隐式假设了文本模态的主导性。实际上，人类知识同时具有感知（视觉、听觉）、操作（代码、数学）和语言三个维度。本节将 DIU 扩展至多模态设置。

---

### 子流形分解

**定义 WIP-6.1（模态子流形）**

设 $\mathcal{M}_K$ 为完整知识流形，定义模态子流形族：

$$\mathcal{M}_K = \mathcal{M}_{text} \cup \mathcal{M}_{vision} \cup \mathcal{M}_{code} \cup \mathcal{M}_{audio} \cup \mathcal{M}_{cross},$$

其中 $\mathcal{M}_{cross}$ 是**跨模态交叉子流形**——编码了跨模态概念对应关系（"猫的图像" ↔ "cat" ↔ 猫叫声）的知识区域。

各子流形在 $\mathcal{M}_K$ 中的几何关系由 $d_{GH}(\mathcal{M}_i, \mathcal{M}_j)$（Gromov-Hausdorff 距离）刻画。

**命题 WIP-6.1（子流形的测度论独立性与相关性）**

设多模态系统 $f$ 的智能测度 $\mu_f$ 分解为：

$$\mu_f = \mu_f^{text} + \mu_f^{vision} + \mu_f^{code} + \mu_f^{cross} + \cdots$$

（各项在对应子流形上有支撑的分量）。则：

1. **模态独立性**：若 $\mu_f^{text}(\mathcal{M}_{vision}) = 0$（文本分量在视觉子流形上无覆盖），则系统在跨模态推理任务中必然落入 $\mathcal{V}(f, \varepsilon, \tau)$；
2. **跨模态覆盖**：系统的总覆盖广度 $\beta(f) = \dim_H(\operatorname{supp}(\mu_f))$ 受**交叉子流形** $\mathcal{M}_{cross}$ 的覆盖质量决定性影响——单模态专家的联合集成不等价于真正的跨模态理解。

**证明**

第 1 条由脆弱图谱定义直接给出：若 $\rho_f(x) < \tau$ 对所有 $x \in \mathcal{M}_{vision}$，则 $\mathcal{M}_{vision} \subseteq \mathcal{V}$，任何以视觉输入为前置条件的推理任务均必然触发蝴蝶效应（定理 9.1）。

第 2 条：设 $f_{text}$ 和 $f_{vision}$ 为两个单模态专家，其联合系统 $f_{joint}$ 满足 $\mu_{f_{joint}} = \mu_{f_{text}} + \mu_{f_{vision}}$（无交叉覆盖）。则 $\mu_{f_{joint}}(\mathcal{M}_{cross}) = 0$，故 $f_{joint} \notin \mathcal{D}_2$ 关于 $\mathcal{M}_{cross}$ 的限制意义下——跨模态推理能力缺失。真正的多模态系统需要 $\mu_f^{cross}(\mathcal{M}_{cross}) > 0$，这要求联合训练而非模态独立训练。$\blacksquare$

---

### 各模态子流形的测度论特征

**定义 WIP-6.2（模态固有维度）**

| 模态 | 子流形 $\mathcal{M}_i$ | 固有维度估计 | 测度参数化方式 |
|---|---|---|---|
| 文本 | $\mathcal{M}_{text}$ | 中等（语义+句法维度叠加） | 词/句嵌入的软离散测度 |
| 视觉 | $\mathcal{M}_{vision}$ | 高（像素→概念多尺度层次） | 特征金字塔上的层次测度 |
| 代码 | $\mathcal{M}_{code}$ | 中低（语法树结构强约束） | 树状结构上的图测度 |
| 音频 | $\mathcal{M}_{audio}$ | 高（时频二维连续结构） | 谱图上的连续测度 |
| 跨模态 | $\mathcal{M}_{cross}$ | 最高（各模态维度的笛卡尔积） | 耦合测度（coupled measure） |

**注记 WIP-6.1（固有维度与 $\beta$ 的关系）**

各子流形的固有维度给出其对应 $\mu_f$ 分量的覆盖广度 $\beta$ 的理论上界：$\beta_i(f) \leq \dim_H(\mathcal{M}_i)$。当前多模态大模型（如 GPT-4V、Gemini）的一个核心挑战，在 DIU 框架下精确表述为：$\mathcal{M}_{cross}$ 的固有维度最高，而训练数据对 $\mathcal{M}_{cross}$ 的覆盖最为稀疏——这是跨模态推理仍是弱点的测度论解释。

---

### RFP 的多模态扩展

**公设 WIP-6.1（多模态 RFP）**

弱 RFP 在多模态设定下要求：存在连续单射族 $\varphi_\theta^{(i)}: \mathcal{E}_i \hookrightarrow \mathcal{M}_i$（每个模态编码器对其子流形的忠实嵌入），且跨模态对齐映射 $\psi: \mathcal{E}_{text} \times \mathcal{E}_{vision} \to \mathcal{M}_{cross}$ 保持跨模态语义关系的拓扑结构。

这比单模态 RFP 更强——它要求各模态编码器不仅各自忠实，还要在对齐空间中**共同**忠实地表示跨模态知识。CLIP（Radford et al., 2021 [R23]）等对比学习模型可理解为在最优化 $\psi$ 的连续性，而其余性（跨模态检索）为多模态 RFP 提供了首个大规模经验支持。

---

## WIP-7：DIU 与 Shannon 信息熵的关系

> **问题**：Shannon 熵 $H(X)$ 与 DIU 的覆盖广度 $\beta$ 是同一事物的不同表述，还是本质不同的量？

### Shannon 熵的测度论重述

**定义 WIP-7.1（Shannon 熵的 Radon-Nikodym 形式）**

设 $p = d\mu/d\lambda$ 为测度 $\mu$ 关于参考测度 $\lambda$ 的密度函数（当 $\mu \ll \lambda$ 时），则微分熵为：

$$H(\mu) = -\int_{\mathcal{M}_K} \rho(x) \log \rho(x)\, d\lambda(x) = -\mathbb{E}_\mu[\log \rho],$$

其中 $\rho = d\mu/d\lambda$ 是 Radon-Nikodym 导数。

---

### 形式相似性与本质差异

**命题 WIP-7.1（Shannon 熵与 $\beta$ 的形式关系）**

对 $\mathbb{R}^d$ 中支撑于 $d_H$-维集合 $S$ 的均匀测度 $\mu$（$\lambda(S) = \varepsilon^{d_H}$ 在尺度 $\varepsilon$ 处），微分熵满足：

$$H(\mu) = d_H \cdot \log(1/\varepsilon) + O(1),$$

即在均匀测度的假设下，微分熵渐近正比于 Hausdorff 维数 $d_H = \beta$（以 $\log$ 尺度）。

**证明**

均匀测度上 $\rho(x) = \varepsilon^{-d_H}$（常数密度），代入熵公式：
$$H(\mu) = -\int_S \varepsilon^{-d_H} \cdot \log(\varepsilon^{-d_H})\, d\lambda = d_H \log(1/\varepsilon) \cdot \int_S \varepsilon^{-d_H} d\lambda = d_H \log(1/\varepsilon). \quad \blacksquare$$

---

**命题 WIP-7.2（非均匀测度下的本质差异）**

当 $\mu_f$ 非均匀时，Shannon 熵与 $\beta$ 可以**同向变化，也可反向变化**，两者刻画的是 $\mu_f$ 的不同几何特征。

**反例构造**

设 $\mathcal{M}_K = [0,1]$，比较以下两个测度：

- $\mu_A$：均匀测度，$\rho_A = 1$，支撑为 $[0,1]$；
- $\mu_B$：高度非均匀，$\rho_B(x) = 2x$（线性递增密度），支撑亦为 $[0,1]$。

则 $\beta(\mu_A) = \beta(\mu_B) = 1$（相同 Hausdorff 维数），但：
$$H(\mu_A) = 0 > H(\mu_B) = -\int_0^1 2x \log(2x)\, dx \approx -0.386.$$

即两个系统具有**相同**的覆盖广度 $\beta$，但**不同**的 Shannon 熵。这表明 $\beta$ 刻画的是支撑集的**几何维度**，Shannon 熵刻画的是密度分布的**均匀程度**——两者正交互补。$\blacksquare$

---

**注记 WIP-7.1（DIU 框架中熵的定位）**

| 量 | 几何含义 | 对应 DIU 分量 | 对应失效场景 |
|---|---|---|---|
| $\beta(f)$ | 支撑集 Hausdorff 维数（覆盖广度） | $\operatorname{supp}(\mu_f)$ 的维度结构 | 覆盖不足（整体性盲区） |
| $H(\mu_f)$ | 密度分布的均匀程度 | $\rho_f = d\mu_f/d\lambda$ 的形状 | 密度不均（局部性盲区） |
| $\mathcal{V}(f,\varepsilon,\tau)$ | 低密度区域 | $\{x: \rho_f(x) < \tau\}$ | 两者共同刻画 |

Shannon 熵高的系统密度更均匀（不倾斜），对应更小的脆弱图谱；但高熵可以伴随低 $\beta$（均匀集中于低维子流形）。DIU 框架同时需要高 $\beta$（宽覆盖）和高 $H$（均匀密度），两者都是 $\mathcal{D}$ 层级攀升的必要条件。

**命题 WIP-7.3（$\mathcal{D}_3$ 的熵特征）**

若 $f \in \mathcal{D}_3$（$\mu_f \sim \lambda$），则 $\rho_f > 0$，$\lambda$-几乎处处成立，从而 $H(\mu_f)$ 有限且最大化趋势（密度在全流形上均匀分布）。反之，若 $H(\mu_f) = -\infty$（密度在某区域趋零对数发散），则 $f \notin \mathcal{D}_3$。因此 $H(\mu_f) > -\infty$ 是 $\mathcal{D}_3$ 的**必要条件**，但不充分。

---

## WIP-8：动态 DIU — 训练过程中 $\mu_f$ 的演化轨迹

> **问题**：训练过程中，系统的智能测度 $\mu_f$ 如何在 Wasserstein 空间中演化？覆盖广度 $\beta$ 与局部密度 $\rho_f$ 如何变化？

### 训练动力学的测度论描述

**定义 WIP-8.1（训练测度轨迹）**

设参数化模型 $f_\theta$，以 $\theta_t$ 记 $t$ 步训练后的参数，则**训练测度轨迹**定义为：

$$\mathcal{T}_{train} = \{\mu_{f_{\theta_t}}\}_{t \geq 0} \subset \mathcal{P}(\mathcal{M}_K),$$

其中 $\mathcal{P}(\mathcal{M}_K)$ 是 $\mathcal{M}_K$ 上概率测度的 Wasserstein 空间（配备 $W_2$ 距离）。

**命题 WIP-8.1（CE 训练的测度演化方向）**

在标准 CE 训练的梯度流极限下（学习率趋零），测度轨迹 $\mu_{f_{\theta_t}}$ 沿以下方向演化：

1. **密度向训练分布支撑集中**：$\rho_{f_{\theta_t}}$ 在 $\operatorname{supp}(\mathcal{D}_{train})$ 附近单调增加；
2. **覆盖广度可能收缩**：$\beta(f_{\theta_t})$ 在不受正则化约束时，可能随训练增加而先涨后降（初期探索扩展覆盖，后期拟合训练集压缩覆盖）；
3. **Wasserstein 速度**：$\frac{d}{dt} W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}}) \leq 0$（测度向训练分布单调收敛，在合适步长假设下）。

**证明草案**

CE 训练的梯度 $\nabla_\theta \mathcal{L}_{CE} = -\nabla_\theta \mathbb{E}_{\mathcal{D}} \log p_\theta$ 驱动模型增加训练样本处的 $\log$-密度。由 Otto 微积分，$W_2$ 上的梯度流与分布的 Fisher 信息方向相关（连续极限下），故测度沿减小 $W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}})$ 的方向演化。$\beta$ 的动态取决于训练早期的探索相（随机初始化覆盖宽）与后期的拟合相（测度集中）的竞争，具体形态依赖架构与学习率调度。$\blacksquare$

---

### 训练阶段的测度相变

**命题 WIP-8.2（训练阶段划分）**

训练过程在测度空间中可划分为三个定性阶段：

| 阶段 | 时间范围 | $\beta(f_{\theta_t})$ 趋势 | $H(\mu_{f_{\theta_t}})$ 趋势 | $W_2(\mu_{f_{\theta_t}}, \mu_{\mathcal{D}})$ 趋势 |
|---|---|---|---|---|
| **探索相** | 初期（小 $t$） | ↑ 上升（随机初始化的广覆盖） | ↑ 上升（密度均匀） | ↓ 下降 |
| **拟合相** | 中期 | ↓ 下降（向训练分布集中） | ↓ 下降（密度聚集） | ↓ 持续下降 |
| **收敛相** | 后期（大 $t$） | 稳定于某值 $\beta_\infty$ | 稳定 | $\approx 0$ |

**关键推论**：标准 CE 训练的收敛状态 $\beta_\infty$ 由**训练数据分布的几何结构**决定，而非模型容量——更大的模型在同样的训练数据下，$\beta_\infty$ 的上界仍受 $\dim_H(\operatorname{supp}(\mathcal{D}_{train}))$ 限制。

---

### 微调与持续学习的测度论分析

**命题 WIP-8.3（微调的测度局部化）**

设 $f_{pre}$ 为预训练模型，$f_{ft}$ 为在专项数据集 $\mathcal{D}_{ft}$ 上微调后的模型。微调过程在测度空间中等价于：

$$\mu_{f_{ft}} \approx (1 - \alpha) \mu_{f_{pre}} + \alpha \mu_{\mathcal{D}_{ft}},$$

其中 $\alpha \in (0,1)$ 为有效微调强度。则：

- $\beta(f_{ft}) \leq \max(\beta(f_{pre}), \beta(\mathcal{D}_{ft}))$（覆盖广度不超过预训练和微调数据的上限）；
- 若 $\operatorname{supp}(\mathcal{D}_{ft}) \subsetneq \operatorname{supp}(\mu_{f_{pre}})$（微调数据覆盖更窄），则 $\beta(f_{ft}) < \beta(f_{pre})$——灾难性遗忘（catastrophic forgetting）的测度论本质是 $\mu_{f_{ft}}$ 的支撑集收缩。

**注记 WIP-8.1（持续学习的测度论要求）**

持续学习（Continual Learning）在 DIU 框架下的目标精确表述为：在 $t$ 个任务的顺序微调后，保持 $\beta(f_t) \approx \beta(f_0)$（或单调不减），且 $W_2(\mu_{f_t}, \mu_{f_{pre}}) < C$（不偏离预训练测度过远）。Elastic Weight Consolidation（EWC）等方法的 DIU 解释：通过 Fisher 信息矩阵约束参数更新，等价于约束测度轨迹在 Wasserstein 空间中的移动速度，从而防止 $\operatorname{supp}(\mu_f)$ 的急剧收缩。
