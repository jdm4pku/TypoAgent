#!/bin/bash
# RQ5 (Scalability and Data Sensitivity): How does the size of induction data affect the performance of TypoAgent?
# Five complete runs: sampling mode fixed, sampling_k values: 5, 10, 15, 20, 25
# All intermediate files, tree files, conversations, and metrics are saved separately by k
#
# Please set API Key before use: export OPENAI_API_KEY=sk-xxx

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: please set the OPENAI_API_KEY environment variable, e.g.: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

DATA_DIR="$REPO/TypoAgent/data"
OUTPUT_BASE="$REPO/output"

echo "=============================================="
echo "RQ5: Scalability and Data Sensitivity"
echo "Impact of sampling_k on TypoAgent performance"
echo "=============================================="

for k in 5 10 15 20; do
  echo ""
  echo "=============================================="
  echo "[k=$k] Complete process 1/2: TypoBuilder (fixed, sampling_k=$k)"
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
  echo "[k=$k] Complete process 2/2: TypoAgent"
  echo "=============================================="

  python run_typoagent.py \
    --tree-path "$OUTPUT_BASE/save_tree_exp5_k${k}/Typo_Tree.json" \
    --mode sample \
    --output-conversation-dir "$OUTPUT_BASE/conversation_exp5_k${k}" \
    --output-metrics-dir "$OUTPUT_BASE/metrics_exp5_k${k}"

  echo ""
  echo "[k=$k] Completed."
done

echo ""
echo "=============================================="
echo "RQ5 completed. Paths for each k:"
echo "  Sampled data: $DATA_DIR/train_new_exp5_k{5,10,15,20,25}.jsonl"
echo "  Tree dir:     $OUTPUT_BASE/save_tree_exp5_k{5,10,15,20,25}/"
echo "  Conversations: $OUTPUT_BASE/conversation_exp5_k{5,10,15,20,25}/"
echo "  Metrics:      $OUTPUT_BASE/metrics_exp5_k{5,10,15,20,25}/"
echo "=============================================="
