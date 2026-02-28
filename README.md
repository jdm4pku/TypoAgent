# OntoAgent

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.26%2B-green.svg)](https://gymnasium.farama.org/)

## 📖 Introduction

OntoAgent is an Ontology-guided Requirements Elicitation Interview Agent.  
This repository provides scripts and configurations to run OntoAgent, its baselines, and ablation studies under a unified experimental framework.

### Key Features

- **Ontology-guided interview**: Uses an ontology tree to guide requirement elicitation conversations.
- **Rich baselines**: Includes Long/Short baselines and mistake-guided variants for comprehensive comparison.
- **Modular ablation**: Supports fine-grained ablation of components such as `ScoreOnto`, `GatePrune`, and `ReRankOnto`.
- **Multi-model support**: Easy to test different LLM backends (e.g., qwen, gpt, gemini) via the same interface.
- **Scalable experiments**: Experiments on different induction data scales (sampling_k) to study scalability.

## 📁 Project Structure

```text
TreeRec/
├── TreeRec/                      # Core modules
│   ├── tree_builder.py          # Semantic tree builder
│   ├── tree_retriever.py        # Tree retriever
│   ├── cluster_tree_builder.py  # Cluster-based tree builder
│   ├── RetrievalAugmentation.py # Main retrieval augmentation class
│   ├── EmbeddingModels.py       # Embedding model wrappers
│   ├── RerankModels.py          # Reranking model wrappers
│   ├── SummarizationModels.py   # Summarization model wrappers
│   ├── Retrievers.py            # Retriever base classes
│   └── utils.py                 # Utility functions
├── IntentRecBench/              # Benchmark datasets and evaluation
│   ├── data/                    # Datasets
│   │   ├── js/                  # JavaScript ecosystem data
│   │   ├── hf/                  # HuggingFace ecosystem data
│   │   └── linux/               # Linux ecosystem data
│   ├── src/                     # Source code
│   │   ├── baselines/           # Baseline method implementations
│   │   └── data_construction/   # Data construction scripts
│   └── script/                  # Execution scripts
├── prompt/                      # Prompt templates
│   ├── summarization_system.txt # Summarization system prompt
│   ├── summarization_user.txt   # Summarization user prompt
│   ├── rerank_system.txt        # Reranking system prompt
│   └── rerank_user.txt          # Reranking user prompt
├── output/                      # Output results directory
├── exp_figure/                  # Experiment figure generation scripts
├── run_treerec.py              # Main execution script
├── run_treerec.sh              # Shell execution script
├── case_study.py               # Case study script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Configure API Key

The project relies on LLM API calls (OpenAI-compatible interface) and requires an API key. Two methods are supported:

- **Method 1: Environment Variables (Recommended)**

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

- **Method 2: Command-line Arguments**

  Some Python scripts support passing `--api-key` and `--base-url` via command line, for example:

  ```bash
  python run_typoagent.py --api-key sk-xxx --base-url https://your-api-endpoint/v1 --mode top3
  ```

> **Note**: Shell scripts under `run_exp/` rely on the `OPENAI_API_KEY` environment variable. If not set, they will fail and prompt you to configure it.

### 2. Run Main Experiment

Run Long Baseline, Short Baseline, Mistake-Guided Baseline, and OntoAgent sequentially:

```bash
cd TypoAgent
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp1.sh
```

## 🧪 Reproducing Experiments

### Using Provided Scripts (`run_exp/`)

All experiment scripts are located in the `run_exp/` directory. **Please configure `OPENAI_API_KEY` before use**.

#### Exp1: Main Experiment Comparison

Run Long Baseline, Short Baseline, Mistake-Guided Baseline, and OntoAgent sequentially:

```bash
cd TypoAgent
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp1.sh
```

#### Exp2: Ablation Study

Run 5 ablation combinations sequentially to validate each module's contribution:

| Order | Combination Name   | Description                               |
|-------|--------------------|-------------------------------------------|
| 1/5   | dfs                | Only Ontology                             |
| 2/5   | dfs_init           | Ontology + ScoreOnto                      |
| 3/5   | dfs_init_gate      | Ontology + ScoreOnto + GatePrune         |
| 4/5   | dfs_init_ctx       | Ontology + ScoreOnto + ReRankOnto        |
| 5/5   | dfs_init_ctx_gate  | Full method                               |

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp2.sh
```

#### Exp4: Multi-Model Testing

Run OntoAgent on multiple models (qwen, gpt, gemini):

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp4.sh
```

#### Exp5: Scalability Experiment (RQ5)

Study the impact of induction data scale (sampling_k) on OntoAgent performance.  
Runs the full pipeline (OntoBuilder + OntoAgent) sequentially for `k=5, 10, 15, 20`:

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp5.sh
```

Outputs are saved to:

- Sampled data: `TypoAgent/data/train_new_exp5_k{5,10,15,20,25}.jsonl`
- Tree directory: `output/save_tree_exp5_k{5,10,15,20,25}/`
- Conversations: `output/conversation_exp5_k{5,10,15,20,25}/`
- Metrics: `output/metrics_exp5_k{5,10,15,20,25}/`

### Running Scripts Individually

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

### Output Directory Structure

- `output/conversation/`: Conversation records
- `output/metrics/`: Evaluation metrics
- `output/save_tree/`: Ontology tree files
