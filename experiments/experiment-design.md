# DIU 实验设计方案 v0.1

## 核心假设

**H1（主假设）**：β_proxy（覆盖广度代理指标）对复杂任务表现的预测力，显著强于 MMLU 平均分。

**H2（蝴蝶效应）**：模型在 MMLU 稀疏子类别的准确率，能预测其在跨域任务上的崩溃位置。

---

## 实验模型（6个）

| 模型 | 规模 | 家族 | Q4 显存估算 |
|---|---|---|---|
| Mistral 7B | 7B | Mistral | ~5 GB |
| Llama 3.1 8B | 8B | Meta | ~5 GB |
| Qwen2.5 7B | 7B | Alibaba | ~5 GB |
| Qwen2.5 32B | 32B | Alibaba | ~20 GB |
| Llama 3.1 70B | 70B | Meta | ~40 GB |
| Qwen2.5 72B | 72B | Alibaba | ~42 GB |

M5 128GB 可一次跑一个模型，顺序执行即可。

---

## Phase 1：MMLU 覆盖地图（计算 β_proxy）

### 方法

用 lm-eval-harness 跑 MMLU 全部 57 个子类别，记录每个模型在每个子类别的准确率。

**β_proxy 定义**：
```
β_proxy(M) = |{ s ∈ MMLU_subjects : acc(M, s) ≥ θ }|
```
其中阈值 θ = 0.60（60%）。

β_proxy 取值范围：0 ~ 57，越高代表覆盖越广。

### 附加指标

同时记录每个模型的**稀疏子类别集合**：
```
Sparse(M) = { s : acc(M, s) < θ }
```
用于 Phase 2 的跨域任务定位。

### 执行命令

```bash
# 安装
pip install lm-eval

# 对每个模型运行（Ollama serving）
lm_eval --model local-completions \
        --model_args base_url=http://localhost:11434/v1,model=llama3.1:8b \
        --tasks mmlu \
        --output_path results/mmlu_llama31_8b \
        --log_samples
```

---

## Phase 2：复杂任务测试

### 数据集选择理由

| 数据集 | 特点 | 获取方式 |
|---|---|---|
| **BIG-Bench Hard (BBH)** | 23个专门挑战LLM的推理任务，已集成lm-eval | `--tasks bbh` |
| **LiveBench** | 每月更新，无训练污染，覆盖数学/代码/推理/语言 | 官方脚本 |

### 执行命令

```bash
# BBH（直接用 lm-eval）
lm_eval --model local-completions \
        --model_args base_url=http://localhost:11434/v1,model=llama3.1:8b \
        --tasks bbh \
        --output_path results/bbh_llama31_8b

# LiveBench（独立脚本）
git clone https://github.com/LiveBench/LiveBench
cd LiveBench
python gen_api_answer.py --model ollama/llama3.1:8b
python show_livebench_result.py
```

---

## Phase 3：分析

### 核心分析

对6个模型，构建如下数据表：

| 模型 | MMLU 平均分 | β_proxy | BBH 分数 | LiveBench 分数 |
|---|---|---|---|---|
| Mistral 7B | ... | ... | ... | ... |
| ... | | | | |

**计算两组相关系数**：
1. `corr(MMLU_avg, BBH)` — 传统指标的预测力
2. `corr(β_proxy, BBH)` — DIU 指标的预测力

**H1 成立条件**：`corr(β_proxy, BBH) > corr(MMLU_avg, BBH)`

### 蝴蝶效应验证

找出 MMLU 平均分相近但 β_proxy 差距较大的两个模型，比较它们在 BBH 跨域任务上的分数差距。

预期结论：β_proxy 更高的模型在跨域任务上显著更好，即使平均分相似。

---

## 预期输出

1. 6×57 的 MMLU 子类别准确率矩阵
2. 每个模型的 β_proxy 值和稀疏子类别列表
3. 6个模型在 BBH 和 LiveBench 上的分数
4. 相关系数对比图（β_proxy vs MMLU_avg 预测力）
5. 稀疏区域热力图（可视化哪些子类别最容易崩溃）

---

## 环境配置

```bash
# Mac M5 推荐配置
brew install ollama
ollama serve &

# 拉取模型（逐个）
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b
ollama pull llama3.1:70b
ollama pull qwen2.5:72b

# Python 环境
pip install lm-eval pandas matplotlib scipy
```

---

## 时间估算

| 阶段 | 每模型耗时 | 6模型合计 |
|---|---|---|
| MMLU（57子类） | 2~4小时（7B），8~12小时（70B） | ~3天 |
| BBH | 1~2小时（7B），4~6小时（70B） | ~1天 |
| LiveBench | 2~3小时（7B），6~8小时（70B） | ~1.5天 |
| **总计** | | **~5~6天** |

建议：大模型（70B/72B）跑夜间批次。

---

*实验设计 v0.1 — 2026-04-21*
