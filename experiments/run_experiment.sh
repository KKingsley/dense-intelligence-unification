#!/bin/bash
# DIU 实验自动化脚本
# 运行环境：Mac M5，需要 Ollama 已启动，lm-eval 已安装
# 用法：bash run_experiment.sh

set -e
RESULTS_DIR="experiments/results"
mkdir -p "$RESULTS_DIR"

MODELS=(
  "mistral:7b"
  "llama3.1:8b"
  "qwen2.5:7b"
  "qwen2.5:32b"
  "llama3.1:70b"
  "qwen2.5:72b"
)

OLLAMA_BASE="http://localhost:11434/v1"

# ── 确认 Ollama 在跑 ──────────────────────────────
if ! curl -s "$OLLAMA_BASE/models" > /dev/null; then
  echo "ERROR: Ollama 未启动，请先运行 'ollama serve'"
  exit 1
fi

echo "======================================"
echo " DIU 实验开始 $(date)"
echo "======================================"

for MODEL in "${MODELS[@]}"; do
  SAFE_NAME="${MODEL//:/_}"
  SAFE_NAME="${SAFE_NAME//\//_}"

  echo ""
  echo ">>> 模型：$MODEL  $(date)"

  # ── MMLU ────────────────────────────────────────
  MMLU_OUT="$RESULTS_DIR/mmlu_${SAFE_NAME}"
  if [ -d "$MMLU_OUT" ]; then
    echo "    MMLU 结果已存在，跳过"
  else
    echo "    运行 MMLU..."
    lm_eval \
      --model local-completions \
      --model_args "base_url=${OLLAMA_BASE},model=${MODEL}" \
      --tasks mmlu \
      --output_path "$MMLU_OUT" \
      --log_samples \
      --batch_size 1
    echo "    MMLU 完成 ✓"
  fi

  # ── BBH ─────────────────────────────────────────
  BBH_OUT="$RESULTS_DIR/bbh_${SAFE_NAME}"
  if [ -d "$BBH_OUT" ]; then
    echo "    BBH 结果已存在，跳过"
  else
    echo "    运行 BBH..."
    lm_eval \
      --model local-completions \
      --model_args "base_url=${OLLAMA_BASE},model=${MODEL}" \
      --tasks bbh \
      --output_path "$BBH_OUT" \
      --batch_size 1
    echo "    BBH 完成 ✓"
  fi

  echo ">>> $MODEL 全部完成 $(date)"
done

# ── 生成汇总 JSON ────────────────────────────────
echo ""
echo ">>> 生成汇总文件..."
python3 - <<'PYEOF'
import json, os, glob

results_dir = "experiments/results"
summary = {}

for model_dir in sorted(glob.glob(f"{results_dir}/mmlu_*")):
    model_name = os.path.basename(model_dir).replace("mmlu_", "")
    result_files = glob.glob(f"{model_dir}/**/*.json", recursive=True)
    if not result_files:
        continue

    with open(result_files[0]) as f:
        data = json.load(f)

    results = data.get("results", {})
    mmlu_scores = {}
    for task, metrics in results.items():
        if task.startswith("mmlu_"):
            subject = task.replace("mmlu_", "")
            mmlu_scores[subject] = metrics.get("acc,none", metrics.get("acc", 0))

    theta = 0.60
    beta_proxy = sum(1 for v in mmlu_scores.values() if v >= theta)
    mmlu_avg = sum(mmlu_scores.values()) / len(mmlu_scores) if mmlu_scores else 0
    sparse = [s for s, v in mmlu_scores.items() if v < theta]

    summary[model_name] = {
        "mmlu_avg": round(mmlu_avg, 4),
        "beta_proxy": beta_proxy,
        "sparse_subjects": sparse,
        "mmlu_by_subject": mmlu_scores,
        "bbh_avg": None  # 后续填入
    }

# 读取 BBH 结果
for model_dir in sorted(glob.glob(f"{results_dir}/bbh_*")):
    model_name = os.path.basename(model_dir).replace("bbh_", "")
    result_files = glob.glob(f"{model_dir}/**/*.json", recursive=True)
    if not result_files or model_name not in summary:
        continue

    with open(result_files[0]) as f:
        data = json.load(f)

    results = data.get("results", {})
    bbh_scores = [v.get("acc,none", v.get("acc", 0))
                  for k, v in results.items() if k.startswith("bbh_")]
    if bbh_scores:
        summary[model_name]["bbh_avg"] = round(sum(bbh_scores)/len(bbh_scores), 4)

with open("experiments/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"汇总完成：{len(summary)} 个模型")
for m, v in summary.items():
    print(f"  {m}: mmlu_avg={v['mmlu_avg']:.3f}, beta_proxy={v['beta_proxy']}, bbh_avg={v['bbh_avg']}")
PYEOF

echo ""
echo "======================================"
echo " 全部完成 $(date)"
echo " 结果：experiments/summary.json"
echo "======================================"

# ── LiveBench（可选，单独运行）──────────────────
# 如果已克隆 LiveBench，取消下面注释：
#
# echo ""
# echo ">>> LiveBench（可选）"
# cd ~/LiveBench
# for MODEL in "${MODELS[@]}"; do
#   python gen_api_answer.py --model "ollama/${MODEL}" \
#     --api-base http://localhost:11434/v1
# done
# python show_livebench_result.py > ~/DIU/experiments/results/livebench_summary.txt
# cd ~/DIU
