#!/bin/bash
# Ablation experiment exp2: sequentially run five ablation configurations
# Before running, please set: export OPENAI_API_KEY=sk-xxx

set -e
cd "$(dirname "$0")/.."

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: please set the OPENAI_API_KEY environment variable, e.g.: export OPENAI_API_KEY=sk-xxx"
  exit 1
fi

echo "=============================================="
echo "Exp2 Ablation Study: sequentially run five ablation configurations"
echo "=============================================="

# 1. DFS traverse all static tree (all disabled)
echo ""
echo "[1/5] Only Ontology"
python run_ablation.py --exp dfs --mode full

# 2. Ontology + ScoreOnto
echo ""
echo "[2/5] Ontology + ScoreOntology"
python run_ablation.py --exp dfs_init --mode full

# 3. Ontology + ScoreOnto + PruneCategory 
echo ""
echo "[3/5] Ontology + ScoreOntology + PruneCategory"
python run_ablation.py --exp dfs_init_gate --mode full

# 4. Ontology + ScoreOnto + ReRankOnto
echo ""
echo "[4/5] Ontology + ScoreOnto + ReRankOnto"
python run_ablation.py --exp dfs_init_ctx --mode full

# 5. Ontology + ScoreOnto + ReRankOnto + PruneCategory
echo ""
echo "[5/5] Ontology + ScoreOnto + ReRankOnto + PruneCategory"
python run_ablation.py --exp dfs_init_ctx_gate --mode full

echo ""
echo "=============================================="
echo "Exp2 Finished"
echo "=============================================="
