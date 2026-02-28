#!/bin/bash
# Experiment exp1: sequentially run the long baseline, short baseline, and TypoAgent
# Before running, please set: export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: please set the OPENAI_API_KEY environment variable, e.g.: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

echo "=============================================="
echo "Exp1: sequentially run long, short, Mistake-Guided, and TypoAgent"
echo "=============================================="

echo ""
echo "[1/4] Baseline Long (LLMREI-Long)"
python run_baselinelong.py --mode full

echo ""
echo "[2/4] Baseline Short (LLMREI-Short)"
python run_baselineshort.py --mode full

echo ""
echo "[3/4] Baseline Mistake-Guided (Mistake-Guided)"
python run_mistakeguided.py --mode full

echo ""
echo "[4/4] TypoAgent"
python run_typoagent.py --mode full

echo ""
echo "=============================================="
echo "Exp1 Finished"
echo "=============================================="
