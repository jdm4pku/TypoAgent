# OntoAgent

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.26%2B-green.svg)](https://gymnasium.farama.org/)

OntoAgent is an Ontology-guided Requirements Elicitation Interview Agent. This document describes how to configure API keys and how to run experiments.

---

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

The project relies on LLM API calls (OpenAI-compatible interface) and requires an API key. Two methods are supported:

#### Method 1: Environment Variables (Recommended)

Set environment variables before running experiments:

```bash
# Required: API Key
export OPENAI_API_KEY=sk-xxx

# Optional: API Base URL (when using third-party proxies or self-hosted services)
# Default: https://api.chatanywhere.tech/v1
export OPENAI_BASE_URL=https://your-api-endpoint/v1
```

To persist configuration using `~/.bashrc`:

```bash
echo 'export OPENAI_API_KEY=sk-xxx' >> ~/.bashrc
source ~/.bashrc
```

#### Method 2: Command-line Arguments

Some Python scripts support passing `--api-key` and `--base-url` via command line, for example:

```bash
python run_typoagent.py --api-key sk-xxx --base-url https://your-api-endpoint/v1 --mode top3
```

> **Note**: Shell scripts under `run_exp/` rely on the `OPENAI_API_KEY` environment variable. If not set, they will fail and prompt you to configure it.

---

## Running Experiments

All experiment scripts are located in the `run_exp/` directory. **Please configure `OPENAI_API_KEY` before use**.

### Exp1: Main Experiment Comparison

Run Long Baseline, Short Baseline, Mistake-Guided Baseline, and OntoAgent sequentially:

```bash
cd TypoAgent
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp1.sh
```

### Exp2: Ablation Study

Run 5 ablation combinations sequentially to validate each module's contribution:

| Order | Combination Name | Description |
|-------|------------------|-------------|
| 1/5 | dfs | Only Ontology  |
| 2/5 | dfs_init | Ontology + ScoreOnto |
| 3/5 | dfs_init_gate | Ontology + ScoreOnto + GatePrune |
| 4/5 | dfs_init_ctx | Ontology + ScoreOnto + ReRankOnto |
| 5/5 | dfs_init_ctx_gate | Full method  |

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp2.sh
```

### Exp4: Multi-Model Testing

Run OntoAgent on multiple models (qwen, gpt, gemini):

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp4.sh
```

### Exp5: Scalability Experiment (RQ5)

Study the impact of induction data scale (sampling_k) on OntoAgent performance. Runs the full pipeline (OntoBuilder + OntoAgent) sequentially for `k=5, 10, 15, 20`:

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp5.sh
```

Outputs are saved to:

- Sampled data: `TypoAgent/data/train_new_exp5_k{5,10,15,20,25}.jsonl`
- Tree directory: `output/save_tree_exp5_k{5,10,15,20,25}/`
- Conversations: `output/conversation_exp5_k{5,10,15,20,25}/`
- Metrics: `output/metrics_exp5_k{5,10,15,20,25}/`

---

## Running Scripts Individually

If not using the scripts in `run_exp/`, you can invoke them manually:

```bash
# OntoAgent (requires existing Onto tree)
python run_typoagent.py --mode top3   # top3 | sample | full | test

# Ablation study
python run_ablation.py --exp dfs_init_ctx_gate --mode top3

# Long/Short Baseline
python run_baselinelong.py --mode top3
python run_baselineshort.py --mode top3

# OntoBuilder (build Onto tree)
python run_typobuilder.py --input TypoAgent/data/train.jsonl --save-dir output/save_tree
```

---

## Output Directory Structure

- `output/conversation/`: Conversation records
- `output/metrics/`: Evaluation metrics
- `output/save_tree/`: Ontology tree files
