#!/bin/bash
# 依次对多个模型运行 TypoAgent
# 使用前请设置：export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: please set the OPENAI_API_KEY environment variable, e.g.: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

for model in qwen gpt gemini claude-opus-4-5-20251101 gemini-3-flash-preview-nothinking deepseek-v3.2 glm-4.7; do
  echo "====== Running model: $model ======"
  python run_typoagent.py --model "$model" --mode full
done