# TypoAgent

基于 LLM 的需求引导与原型生成系统。本文档介绍如何配置 API Key 及如何运行实验。

---

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

项目依赖大语言模型 API 调用（OpenAI 兼容接口），需配置 API Key。支持以下两种方式：

#### 方式一：环境变量（推荐）

在运行实验前设置环境变量：

```bash
# 必需：API Key
export OPENAI_API_KEY=sk-xxx

# 可选：API Base URL（使用第三方代理或自建服务时）
# 默认: https://api.chatanywhere.tech/v1
export OPENAI_BASE_URL=https://your-api-endpoint/v1
```

若使用 `~/.bashrc` 持久化配置：

```bash
echo 'export OPENAI_API_KEY=sk-xxx' >> ~/.bashrc
source ~/.bashrc
```

#### 方式二：命令行参数

部分 Python 脚本支持通过 `--api-key` 和 `--base-url` 传入，例如：

```bash
python run_typoagent.py --api-key sk-xxx --base-url https://your-api-endpoint/v1 --mode top3
```

> **注意**：`run_exp/` 下的 shell 脚本统一依赖环境变量 `OPENAI_API_KEY`，若未设置会报错并提示配置。

---

## 运行实验

所有实验脚本位于 `run_exp/` 目录下。**使用前请先配置 `OPENAI_API_KEY`**。

### Exp1：主实验对比

依次运行 Long Baseline、Short Baseline、TypoAgent：

```bash
cd /home/ubuntu/jdm/xiaotian/TypoAgent_release_v2
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp1.sh
```

### Exp2：消融实验

依次运行 5 种消融组合，验证各模块贡献：

| 顺序 | 组合名称 | 说明 |
|------|----------|------|
| 1/5 | dfs | DFS 遍历全部静态树（全部关闭） |
| 2/5 | dfs_init | DFS + 对初始需求优先度打分排序 |
| 3/5 | dfs_init_gate | DFS + 初始需求优先度 + 大类门控剪枝 |
| 4/5 | dfs_init_ctx | DFS + 初始需求优先度 + 过程中上下文打分排序 |
| 5/5 | dfs_init_ctx_gate | 完整方法（所有模块） |

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp2.sh
```

### Exp4：多模型测试

对多个模型（qwen、gpt、gemini）分别运行 TypoAgent：

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp4.sh
```

### Exp5：可扩展性实验（RQ5）

研究 induction 数据规模（sampling_k）对 TypoAgent 性能的影响。依次对 `k=5, 10, 15, 20, 25` 运行完整流程（TypoBuilder + TypoAgent）：

```bash
export OPENAI_API_KEY=sk-xxx
bash run_exp/run_exp5.sh
```

输出分别保存至：

- 采样数据：`TypoAgent/data/train_new_exp5_k{5,10,15,20,25}.jsonl`
- 树目录：`output/save_tree_exp5_k{5,10,15,20,25}/`
- 对话：`output/conversation_exp5_k{5,10,15,20,25}/`
- 指标：`output/metrics_exp5_k{5,10,15,20,25}/`

---

## 单独运行脚本

若不使用 `run_exp/` 中的脚本，可手动调用：

```bash
# TypoAgent（需已有 Typo 树）
python run_typoagent.py --mode top3   # top3 | sample | full | test

# 消融实验
python run_ablation.py --exp dfs_init_ctx_gate --mode top3

# Long/Short Baseline
python run_baselinelong.py --mode top3
python run_baselineshort.py --mode top3

# TypoBuilder（构建 Typo 树）
python run_typobuilder.py --input TypoAgent/data/train.jsonl --save-dir output/save_tree
```

---

## 输出目录结构

- `output/conversation/`：对话记录
- `output/metrics/`：评估指标
- `output/save_tree/`：Typo 树文件
