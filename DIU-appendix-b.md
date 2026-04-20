# DIU 附录B：实验设计

---

## 附录B：开放问题 2 实验设计

### B.1 核心假设

**H1（广度预测力）** $\beta(f)$ 对复杂多步任务性能的预测力，优于任意有限 benchmark 组合

**H2（脆弱图谱预测力）** $\mathcal{V}(f,\varepsilon,\tau)$ 能在任务失败发生前，预测具体失败位置

**H3（结构距离优越性）** $W_2(\mu_A,\mu_B)$ 对模型间性能差距的解释力，优于任务特定得分差

### B.2 查询集构造

```
|Q| ≥ 10,000，按以下维度分层采样：

域维度（各 ~15%）：数学 / 代码 / 科学推理 /
                    语言理解 / 跨域组合 / 常识 / 元认知

复杂度维度：
  L1 — 单步知识检索      （benchmark 能捕捉）
  L2 — 二步因果推理
  L3 — 多步跨域组合      （benchmark 无法捕捉）
  L4 — 对抗性扰动任务
```

关键设计原则：**L3/L4 任务的路径必须经过子流形的过渡区域**，这是 $\mathcal{V}(f)$ 的感应区。

### B.3 度量计算流程

**Step 1：嵌入收集**
固定参考编码器（E5-mistral-7b），收集 $E_f = \{\phi(\text{response}_f(q_i))\}_{i=1}^N \subset \mathbb{R}^d$

**Step 2：覆盖广度 $\beta(f)$**
$$\hat\beta(f) = \widehat{\dim}_{\text{TwoNN}}(E_f) = \left(\frac{1}{N}\sum_{i=1}^N \log\frac{r_{i,2}}{r_{i,1}}\right)^{-1}$$

**Step 3：结构距离 $W_2$**
$$\widetilde W_2(\hat\mu_A,\hat\mu_B) = \left(\int_{\mathbb{S}^{d-1}} W_2^2(P_\theta\hat\mu_A, P_\theta\hat\mu_B)\,d\theta\right)^{1/2}$$
蒙特卡洛积分近似（~1000 随机方向）

**Step 4：脆弱图谱 $\mathcal{V}$**
$$\hat\rho_f(x_i,k) = \frac{k}{N \cdot V_d \cdot r_{i,k}^d}$$
标记 $\hat{\mathcal{V}}(f,k,\tau) = \{x_i : \hat\rho_f(x_i,k) < \tau\}$，按语义类别聚类

### B.4 验证方案

**实验一：广度与复杂任务性能的相关性**

| 测量 | 期望结果 |
|---|---|
| $\text{Corr}(\beta(f),\; \text{perf}_{L3})$ | $> 0.7$ |
| $\text{Corr}(\text{MMLU},\; \text{perf}_{L3})$ | $< 0.55$ |
| $\text{Corr}(\beta(f),\; \text{perf}_{L1})$ | $\approx \text{Corr}(\text{MMLU},\; \text{perf}_{L1})$ |

**实验二：脆弱图谱对失败模式的预测**

1. 计算各模型 $\hat{\mathcal{V}}(f)$，识别低密度语义聚类
2. 设计专项探针任务集 $\mathcal{Q}_{\mathcal{V}}$：查询点落在 $\hat{\mathcal{V}}$ 区域内
3. 盲测：在看到得分前，根据 $\hat{\mathcal{V}}$ 预测失败率
4. 验证 Lyapunov 放大系数 $\Lambda \approx \log(1/\hat\tau)$

**实验三：Benchmark 污染的 DIU 检测**

$$\text{Contamination Score}(f,\mathcal{B}) = \frac{\hat\rho_f(x_{\mathcal{B}},\varepsilon)}{\beta(f)} - 1$$

显著为正 → 该区域局部密度人工升高 → benchmark 污染信号

### B.5 实验资源

```
模型：GPT-4o, Claude-3.5-Sonnet, Llama-3-70B,
      DeepSeek-V3, Qwen-2.5-72B, Mistral-Large

查询集：10k 条，标注域 + 复杂度层级
参考编码器：E5-mistral-7b（开源，可复现）
计算：嵌入 ~6 GPU-hours；TwoNN/SW 距离 ~CPU 2h
```

### B.6 可证伪条件

若以下任一成立，DIU 核心主张需修正：

- $\text{Corr}(\beta(f), \text{perf}_{L3}) < 0.4$（广度无预测力）
- $\hat{\mathcal{V}}$ 对失败位置的预测精度不优于随机基线
- $\text{Corr}(\text{MMLU}, \text{perf}_{L3}) > 0.85$（benchmark 已足够）
