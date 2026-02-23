#!/bin/bash
# 消融实验 exp2：依次运行 5 种消融组合
# 使用前请设置：export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "错误: 请设置 OPENAI_API_KEY 环境变量，例如: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

echo "=============================================="
echo "Exp2 消融实验：依次运行 5 种组合"
echo "=============================================="

# 1. DFS 遍历全部静态树（全部关闭）
echo ""
echo "[1/5] dfs: DFS 遍历全部静态树"
python run_ablation.py --exp dfs --mode top3

# 2. DFS + 对初始需求优先度打分排序
echo ""
echo "[2/5] dfs_init: DFS + 对初始需求优先度打分排序"
python run_ablation.py --exp dfs_init --mode top3

# 3. DFS + 对初始需求优先度打分排序 + 大类门控剪枝
echo ""
echo "[3/5] dfs_init_gate: DFS + 对初始需求优先度打分排序 + 大类门控剪枝"
python run_ablation.py --exp dfs_init_gate --mode top3

# 4. DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序
echo ""
echo "[4/5] dfs_init_ctx: DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序"
python run_ablation.py --exp dfs_init_ctx --mode top3

# 5. DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序 + 大类门控剪枝（完整方法）
echo ""
echo "[5/5] dfs_init_ctx_gate: DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序 + 大类门控剪枝"
python run_ablation.py --exp dfs_init_ctx_gate --mode top3

echo ""
echo "=============================================="
echo "Exp2 消融实验全部完成"
echo "=============================================="
