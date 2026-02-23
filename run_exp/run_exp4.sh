#!/bin/bash
# 依次对多个模型运行 TypoAgent
# 使用前请设置：export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "错误: 请设置 OPENAI_API_KEY 环境变量，例如: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

for model in qwen gpt gemini; do
  echo "====== Running model: $model ======"
  python run_typoagent.py --model "$model"
done
