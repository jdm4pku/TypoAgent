#!/bin/bash
# RQ5 (Scalability and Data Sensitivity): How does the size of induction data affect the performance of TypoAgent?
# 五次完整流程：采样模式 fixed，sampling_k 分别取 5, 10, 15, 20, 25
# 所有中间文件、树文件、对话、指标均按 k 分别保存
#
# 使用前请设置 API Key：export OPENAI_API_KEY=sk-xxx

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "错误: 请设置 OPENAI_API_KEY 环境变量，例如: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

DATA_DIR="$REPO/TypoAgent/data"
OUTPUT_BASE="$REPO/output"

echo "=============================================="
echo "RQ5: Scalability and Data Sensitivity"
echo "sampling_k 对 TypoAgent 性能的影响"
echo "=============================================="

for k in 5 10 15 20 25; do
  echo ""
  echo "=============================================="
  echo "[k=$k] 完整流程 1/2: TypoBuilder (fixed, sampling_k=$k)"
  echo "=============================================="

  python run_typobuilder.py \
    --enable-sampling \
    --sampling-mode fixed \
    --sampling-k "$k" \
    --input "$DATA_DIR/train.jsonl" \
    --sampled-output "$DATA_DIR/train_new_exp5_k${k}.jsonl" \
    --save-dir "$OUTPUT_BASE/save_tree_exp5_k${k}" \
    --tree-output "$OUTPUT_BASE/save_tree_exp5_k${k}/Typo_Tree.json"

  echo ""
  echo "=============================================="
  echo "[k=$k] 完整流程 2/2: TypoAgent"
  echo "=============================================="

  python run_typoagent.py \
    --tree-path "$OUTPUT_BASE/save_tree_exp5_k${k}/Typo_Tree.json" \
    --mode full \
    --output-conversation-dir "$OUTPUT_BASE/conversation_exp5_k${k}" \
    --output-metrics-dir "$OUTPUT_BASE/metrics_exp5_k${k}"

  echo ""
  echo "[k=$k] 完成."
done

echo ""
echo "=============================================="
echo "RQ5 全部完成. 各 k 对应路径："
echo "  采样数据: $DATA_DIR/train_new_exp5_k{5,10,15,20,25}.jsonl"
echo "  树目录:   $OUTPUT_BASE/save_tree_exp5_k{5,10,15,20,25}/"
echo "  对话:     $OUTPUT_BASE/conversation_exp5_k{5,10,15,20,25}/"
echo "  指标:     $OUTPUT_BASE/metrics_exp5_k{5,10,15,20,25}/"
echo "=============================================="
