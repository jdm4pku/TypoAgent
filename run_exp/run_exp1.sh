#!/bin/bash
# 实验 exp1：依次运行 long baseline、short baseline、TypoAgent
# 使用前请设置：export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "错误: 请设置 OPENAI_API_KEY 环境变量，例如: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

echo "=============================================="
echo "Exp1：依次运行 long、short、TypoAgent"
echo "=============================================="

echo ""
echo "[1/3] Baseline Long (LLMREI-Long)"
python run_baselinelong.py --mode top3

echo ""
echo "[2/3] Baseline Short (LLMREI-Short)"
python run_baselineshort.py --mode top3

echo ""
echo "[3/3] TypoAgent"
python run_typoagent.py --mode top3

echo ""
echo "=============================================="
echo "Exp1 全部完成"
echo "=============================================="
