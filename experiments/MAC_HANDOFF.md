# MAC 实验交接说明

## 你需要做的事：在 Mac M5 上跑完整实验，把结果推回 GitHub

---

## Step 1：环境准备（一次性）

```bash
# 1. 安装 Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装 Ollama
brew install ollama

# 3. 安装 Python 依赖
pip3 install lm-eval pandas matplotlib scipy numpy

# 4. 克隆 LiveBench
git clone https://github.com/LiveBench/LiveBench ~/LiveBench

# 5. 克隆 DIU 仓库（或者你直接拷贝这个文件夹过来也行）
git clone https://github.com/KKingsley/dense-intelligence-unification ~/DIU
cd ~/DIU
```

---

## Step 2：拉取所有模型（需要时间，建议提前跑）

```bash
ollama serve &   # 后台启动

# 按顺序拉取（小模型先，大模型后）
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b
ollama pull llama3.1:70b
ollama pull qwen2.5:72b
```

---

## Step 3：运行全部实验

```bash
cd ~/DIU
bash run_experiment.sh
```

脚本会自动：
- 对6个模型依次跑 MMLU + BBH
- 结果保存在 `experiments/results/` 目录
- 最后生成汇总文件 `experiments/summary.json`

LiveBench 需要单独跑（见脚本内注释）。

---

## Step 4：把结果推回 GitHub

```bash
cd ~/DIU
git add experiments/results/
git add experiments/summary.json
git commit -m "feat(exp): 实验结果 MMLU+BBH 6模型"
git push origin main
```

---

## 你需要回传的文件（必须）

| 文件 | 内容 |
|---|---|
| `experiments/summary.json` | 6模型 × 所有指标汇总 |
| `experiments/results/mmlu_*.json` | 每个模型的 MMLU 子类别详细结果 |
| `experiments/results/bbh_*.json` | 每个模型的 BBH 结果 |

LiveBench 结果如果能跑完也一起推，跑不完没关系，MMLU + BBH 够用。

---

## 注意事项

- 70B/72B 模型建议**夜间跑**，每个约 8~12 小时
- M5 128GB 够用，但一次只跑一个模型，不要并行
- 如果某个模型跑崩了，单独重跑那一个即可，其他结果不受影响
- Ollama 默认端口 11434，脚本已配好，不用改
