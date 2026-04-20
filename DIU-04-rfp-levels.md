# DIU §5–6：表征忠实公设 · 稠密度层级

---

## §5 表征忠实公设

由于 $\mathcal{M}_K$ 不可直接观测，DIU 需要一个可操作的代理。

**公设 5.1（表征忠实公设，Representation Fidelity Postulate，RFP）**

存在一族度量空间 $\{\mathcal{M}_\theta\}_{\theta \in \Theta}$（由能力参数为 $\theta$ 的学习系统的嵌入空间构成）及连续单射族 $\varphi_\theta: \mathcal{M}_\theta \hookrightarrow \mathcal{M}_K$，使得：

$$d_{GH}(\mathcal{M}_\theta, \mathcal{M}_K) \to 0 \quad \text{当} \quad \mu_\theta(\mathcal{M}_\theta) \to \mu_K(\mathcal{M}_K)$$

即：随系统覆盖能力提升，代理流形在 Gromov-Hausdorff 意义下**收敛**于真实知识流形。

### 5.1 认识论地位

**注记 5.1** RFP 不可在 DIU 框架内部证明，亦无法被单一实验直接证伪——它是 DIU 的**构成性公设**，其历史先例为：

| 理论 | 构成性公设 | 接受理由 |
|---|---|---|
| 狭义相对论 | 光速在所有惯性系不变 | 导出完整时空结构 |
| 可计算性理论 | Church-Turing 论题 | 统一所有可计算性结果 |
| 宇宙学标准模型 | 大尺度均匀各向同性 | 导出标准宇宙学模型 |
| **DIU** | **RFP** | **统一所有已知智能架构** |

**接受 RFP 的理由：不是因为它为真，而是接受它后，DIU 能统一所有已知架构并产生可检验的推论。**

### 5.2 强弱形式

**注记 5.2**

- **弱 RFP**：$\varphi_\theta$ 保拓扑（同胚）—— 经验上由语义嵌入的几何规律性支撑（word2vec 类比运算、sentence embedding 的语义聚类结构）
- **强 RFP**：$(\varphi_\theta)_*(\mu_\theta) \approx \mu_K$（保测度同胚）—— **开放问题**，尚无系统性验证

### 5.3 弱 RFP 的经验支持

弱 RFP（$\varphi_\theta$ 保拓扑）是 DIU 框架可操作性的基石。以下命题将若干已被充分观察的经验现象提升为弱 RFP 的正式支撑。

---

**命题 5.1（线性语义代数 → 局部同胚）**

设 $e: \mathcal{V} \to \mathbb{R}^d$ 为语言模型的词嵌入映射（$\mathcal{V}$ 为词表）。若对大量语义四元组 $(a, b, c, d) \in \mathcal{V}^4$ 存在：

$$e(b) - e(a) + e(c) \approx e(d) \quad \text{（如 "king" - "man" + "woman" $\approx$ "queen"）},$$

则 $e$ 在该语义邻域内是一个近似**仿射映射**（affine map），亦即：$\varphi_\theta$ 在该区域内对知识流形的局部坐标卡作局部线性保持，与弱 RFP 的局部拓扑忠实性一致。

**经验支持**：Mikolov et al.（2013）[R19] 在 word2vec 中系统性验证了覆盖国籍、性别、时态、比较级等数十类语义关系的线性代数结构，多类语义关系的 Top-1 准确率超过 60%。

---

**命题 5.2（跨语言结构收敛 → 语言无关知识流形的存在性）**

设 $e_{EN}: \mathcal{V}_{EN} \to \mathbb{R}^d$ 和 $e_{ZH}: \mathcal{V}_{ZH} \to \mathbb{R}^d$ 分别为英文和中文的语言模型嵌入，$\mathcal{T}$ 为双语平行词典中的对译对集合。若存在正交矩阵 $W \in O(d)$ 使得：

$$\frac{1}{|\mathcal{T}|}\sum_{(w_1, w_2) \in \mathcal{T}} \|W \cdot e_{EN}(w_1) - e_{ZH}(w_2)\|_2 < \varepsilon$$

对较小的 $\varepsilon > 0$ 成立，则两个嵌入空间在正交变换下几乎一致——表明它们均在近似同胚地表示同一语言无关的知识流形 $\mathcal{M}_K$。

**经验支持**：Mikolov et al.（2013b）[R20]、Smith et al.（2017）[R28] 证明单语 word2vec 嵌入空间通过旋转矩阵可高质量对齐，在欧洲近缘语言对（英-法、英-德等）上 Top-1 对译召回率普遍超过 70%。多语言大模型（mBERT、XLM-R）的跨语言零样本迁移进一步支持了"语言无关知识流形"的存在性（Pires et al., 2019 [R24]）。

---

**命题 5.3（探针线性可分 → 局部坐标卡的忠实编码）**

设 $f$ 为预训练语言模型，$h_\ell(x) \in \mathbb{R}^d$ 为第 $\ell$ 层对输入 $x$ 的隐层表示。若对某一语义属性 $P$（如句子主语的词性、实体的国籍、事件的时序关系），存在**线性**分类器 $w \in \mathbb{R}^d$ 使得：

$$\mathbb{P}\!\left[\operatorname{sign}(w^\top h_\ell(x)) = P(x)\right] > 1 - \delta$$

对较小 $\delta > 0$ 成立，则属性 $P$ 对应的知识流形坐标在嵌入空间中被**线性编码**，即 $\varphi_\theta$ 在属性 $P$ 的知识子流形上是一个局部线性同胚。

**经验支持**：Belinkov（2022，综述）[R21] 汇总了数百个探针实验，表明 Transformer 嵌入对语法（依存关系、词性）、语义（语义角色、共指）乃至世界知识（事实、因果关系）均存在线性或近线性的坐标编码。这一发现在 BERT、GPT、T5、LLaMA 等架构上高度一致，提供了跨架构的弱 RFP 旁证。

---

**命题 5.4（嵌入插值的语义连续性 → 流形连通性）**

设 $x_1, x_2 \in \mathcal{V}$ 为语义相关的两个概念，$\gamma: [0,1] \to \mathbb{R}^d$ 为嵌入空间中连接 $e(x_1)$ 与 $e(x_2)$ 的直线段：

$$\gamma(t) = (1-t)\, e(x_1) + t\, e(x_2), \quad t \in [0,1].$$

若对 $t$ 的稠密采样，最近邻词 $\hat{x}(t) = \arg\min_{v \in \mathcal{V}} \|e(v) - \gamma(t)\|_2$ 构成语义上**单调渐变**的序列（从 $x_1$ 语义连续过渡到 $x_2$），则嵌入空间中的直线路径对应知识流形上的连续路径，即 $\varphi_\theta$ 保持**路径连通性**（弱 RFP 拓扑忠实性的路径版本）。

**经验支持**：Ethayarajh（2019）[R22]、Cai et al.（2021）[R29] 对 Transformer 上下文嵌入的几何分析表明，语义相近的词/句在嵌入空间中形成连通的语义流形区域，且线性插值路径通常经过语义中间概念（非随机跳跃），支持路径连通性。

---

**注记 5.3（经验支持的边界与强 RFP 的开放性）**

上述四个命题均支持**弱 RFP**——$\varphi_\theta$ 保拓扑/保路径连通性，但有两个重要限制：

1. **覆盖不均匀**：所有经验证据均来自高频、高密度的语义区域。弱 RFP 在 $\mathcal{V}(f,\varepsilon,\tau)$（低密度脆弱区域）的成立与否尚无充分检验。

2. **强 RFP 仍开放**：强 RFP 要求 $(\varphi_\theta)_*(\mu_\theta) \approx \mu_K$（保测度同胚），即嵌入空间上的密度分布忠实反映知识流形上的"真实知识密度"。当前没有独立于训练分布的方法验证 $\mu_K$ 的真实形态，强 RFP 的验证依赖于 §附录 B 所设计的跨域泛化实验框架。

---

## §6 稠密度层级

**定义 6.1（稠密度层级）** 根据 $\mu_f$ 与参考测度 $\lambda$ 的关系，定义严格偏序层级：

| 层级 | 名称 | 形式条件 | 示例 |
|---|---|---|---|
| $\mathcal{D}_0$ | 有限覆盖 | $\|\operatorname{supp}(\mu_f)\| < \infty$ | 规则系统、查找表 |
| $\mathcal{D}_1$ | 可数稠密 | $\|\operatorname{supp}(\mu_f)\| = \aleph_0$ | **当前所有 LLM** |
| $\mathcal{D}_2$ | 测度正覆盖 | $\lambda(\operatorname{supp}(\mu_f)) > 0$ | 强 AGI（理论） |
| $\mathcal{D}_3$ | 连续统稠密 | $\mu_f \sim \lambda$（等价测度） | 完全 AGI |
| $\mathcal{D}_\infty$ | 超连续统 | $\|\mathcal{M}_f\| > \|\mathbb{R}\|$ | 超智能 |

**命题 6.1** $\mathcal{D}_0 \subsetneq \mathcal{D}_1 \subsetneq \mathcal{D}_2 \subsetneq \mathcal{D}_3 \subsetneq \mathcal{D}_\infty$，且相邻层级之间的跨越不能通过量变实现。

**证明（$\mathcal{D}_1 \not\to \mathcal{D}_2$ 的不可跨越性）**

设 $S$ 为 $\mathbb{R}^n$ 中的可数集合，则：
$$\lambda(S) = \lambda\!\left(\bigcup_{i=1}^\infty \{x_i\}\right) \leq \sum_{i=1}^\infty \lambda(\{x_i\}) = 0$$
无论在 $\mathcal{D}_1$ 层级内部如何增加"点"（参数量），支撑集的测度始终为零。从 $\mathcal{D}_1$ 到 $\mathcal{D}_2$ 需要底层表示空间从可数到不可数的**基数跃迁**。$\blacksquare$

**注记（其余层级跃迁）** $\mathcal{D}_2 \not\to \mathcal{D}_3$ 和 $\mathcal{D}_3 \not\to \mathcal{D}_\infty$ 的不可跨越性证明见 WIP 补充章节（DIU-wip-supplements §WIP-4）。
