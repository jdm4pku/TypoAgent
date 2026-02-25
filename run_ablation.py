#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验入口：通过 --exp 选择实验类型，一步完成读取树、测试、输出实验结果。

消融选项：
- dfs               : DFS 遍历全部静态树（全部关闭）
- dfs_init          : DFS + 对初始需求优先度打分排序
- dfs_init_gate     : DFS + 对初始需求优先度打分排序 + 大类门控剪枝
- dfs_init_ctx      : DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序
- dfs_init_ctx_gate : DFS + 对初始需求优先度打分排序 + 过程中上下文打分排序 + 大类门控剪枝（完整方法）
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def select_samples_by_application_type(all_tasks, samples_per_type=1, seed=None):
    """按 application_type 分组，每个类型选择指定数量的样本。"""
    if seed is not None:
        random.seed(seed)
    tasks_by_type = defaultdict(list)
    for task in all_tasks:
        app_type = task.get("application_type", "Unknown")
        tasks_by_type[app_type].append(task)
    selected_tasks = []
    for app_type, tasks in tasks_by_type.items():
        num_available = len(tasks)
        if num_available >= samples_per_type:
            selected = random.sample(tasks, samples_per_type)
            selected_tasks.extend(selected)
        else:
            selected_tasks.extend(tasks)
    return selected_tasks


def build_parser():
    parser = argparse.ArgumentParser(
        description="ReqElicitGym 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融组合:
  dfs               : DFS 遍历静态树（全部关闭）
  dfs_init          : DFS + 初始需求优先度
  dfs_init_gate     : DFS + 初始需求优先度 + 大类门控剪枝
  dfs_init_ctx      : DFS + 初始需求优先度 + 上下文优先度
  dfs_init_ctx_gate : DFS + 初始需求优先度 + 上下文优先度 + 大类门控剪枝（完整方法）
        """,
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["dfs", "dfs_init", "dfs_init_gate", "dfs_init_ctx", "dfs_init_ctx_gate"],
        default="dfs_init_ctx_gate",
        help="消融组合",
    )
    parser.add_argument(
        "--tree-path",
        type=str,
        default=None,
        help="树文件路径，默认 output/save_tree/LLMTree.auto-p.json",
    )
    parser.add_argument(
        "--mode",
        choices=["top3", "sample", "full", "test"],
        default=None,
        help="top3=前3个, sample=采样N个, full=全部, test=仅第1个任务",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="sample 模式采样数，默认 20",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Interviewer 使用的 LLM 模型",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Interviewer LLM 的 temperature",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="API 调用超时时间（秒），默认 60",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key，也可用 OPENAI_API_KEY",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API Base URL",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge 模型",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default=None,
        help="User Simulator 模型",
    )
    parser.add_argument(
        "--output-conversation-dir",
        type=str,
        default=None,
        help="对话输出目录",
    )
    parser.add_argument(
        "--output-metrics-dir",
        type=str,
        default=None,
        help="指标输出目录",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="最大对话步数",
    )
    parser.add_argument(
        "--max-turns-per-category-no-gate",
        type=int,
        default=None,
        help="关闭大类门控时，每个大类最多提问次数",
    )
    parser.add_argument(
        "--cat-check-threshold",
        type=int,
        default=None,
        help="大类确认阈值：连续多少轮无新增后触发第一次大类确认，默认 2",
    )
    parser.add_argument(
        "--followup-threshold",
        type=int,
        default=None,
        help="大类追问阈值：第一次确认后连续多少轮无新增再触发追问，默认 4",
    )
    parser.add_argument(
        "--tree-percentage",
        type=float,
        default=None,
        help="树裁剪百分比，100.0 表示使用全部",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="关闭详细输出",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # ==================== 手动修改默认配置区域 ====================
    DEFAULTS = {
        "tree_path": str(_REPO_ROOT / "output" / "save_tree" / "Typo_Tree.json"),
        "mode": "test",
        "sample_n": 20,
        "model": "gpt-5.1",
        "temperature": 0.0,
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.chatanywhere.tech/v1",
        "judge_model": "gpt-5.1",
        "user_model": "gpt-5.1",
        "output_conversation_dir": str(_REPO_ROOT / "output" / "conversation"),
        "output_metrics_dir": str(_REPO_ROOT / "output" / "metrics"),
        "data_path": str(_REPO_ROOT / "ReqElicitGym" / "data" / "test.json"),
        "max_steps": 20,
        "max_turns_per_category_no_gate": 3,
        "cat_check_threshold": 2,
        "followup_threshold": 2,
        "tree_percentage": 100.0,
        "timeout": 60.0,
        "verbose": True,
    }
    # ============================================================

    # 解析消融组合
    exp_name = args.exp
    use_initial_priority = exp_name not in ("dfs",)
    use_context_priority = exp_name in ("dfs_init_ctx", "dfs_init_ctx_gate")
    use_category_gating = exp_name in ("dfs_init_gate", "dfs_init_ctx_gate")

    # 应用参数与默认值
    tree_path = args.tree_path or DEFAULTS["tree_path"]
    mode = args.mode or DEFAULTS["mode"]
    sample_n = args.sample_n if args.sample_n is not None else DEFAULTS["sample_n"]
    model = args.model or DEFAULTS["model"]
    temperature = args.temperature if args.temperature is not None else DEFAULTS["temperature"]
    timeout = args.timeout if args.timeout is not None else DEFAULTS["timeout"]
    api_key = args.api_key or DEFAULTS["api_key"]
    base_url = args.base_url or DEFAULTS["base_url"]
    judge_model = args.judge_model or DEFAULTS["judge_model"]
    user_model = args.user_model or DEFAULTS["user_model"]
    conversation_dir = args.output_conversation_dir or DEFAULTS["output_conversation_dir"]
    metrics_dir = args.output_metrics_dir or DEFAULTS["output_metrics_dir"]
    data_path = DEFAULTS["data_path"]
    max_steps = args.max_steps if args.max_steps is not None else DEFAULTS["max_steps"]
    max_turns_per_category_no_gate = args.max_turns_per_category_no_gate if args.max_turns_per_category_no_gate is not None else DEFAULTS["max_turns_per_category_no_gate"]
    cat_check_threshold = args.cat_check_threshold if args.cat_check_threshold is not None else DEFAULTS["cat_check_threshold"]
    followup_threshold = args.followup_threshold if args.followup_threshold is not None else DEFAULTS["followup_threshold"]
    tree_percentage = args.tree_percentage if args.tree_percentage is not None else DEFAULTS["tree_percentage"]
    verbose = DEFAULTS["verbose"] if not args.no_verbose else (args.verbose or False)

    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 或 --api-key")
        sys.exit(1)

    # 确保树路径可访问（支持相对路径）
    if not os.path.isabs(tree_path):
        tree_path = str(_REPO_ROOT / tree_path)
    if not os.path.exists(tree_path):
        candidate = _REPO_ROOT / "output" / "save_tree" / "LLMTree.auto-p.json"
        if candidate.exists():
            tree_path = str(candidate)
        else:
            print(f"错误: 找不到树文件 {tree_path}")
            sys.exit(1)

    # 加载数据并选择任务
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据文件 {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    total_tasks_in_file = len(all_tasks)

    if mode == "test":
        selected_tasks = all_tasks[:1]
        print(f"测试模式: test, 从 {total_tasks_in_file} 条任务中只运行第 1 条")
    elif mode == "top3":
        selected_tasks = all_tasks[:3]
        print(f"测试模式: top3, 从 {total_tasks_in_file} 条任务中取前 3 条")
    elif mode == "full":
        selected_tasks = all_tasks
        print(f"测试模式: full, 使用全部 {total_tasks_in_file} 条任务")
    else:  # sample
        selected_tasks = random.sample(all_tasks, min(sample_n, len(all_tasks)))
        print(f"测试模式: sample, 原始 {total_tasks_in_file} 条, 抽样后 {len(selected_tasks)} 条")

    # 写入临时数据文件
    data_dir = Path(data_path).parent
    temp_name = f"test_ablation_{mode}_{exp_name}.json"
    temp_data_path = data_dir / temp_name
    with open(temp_data_path, "w", encoding="utf-8") as f:
        json.dump(selected_tasks, f, ensure_ascii=False, indent=2)

    os.makedirs(conversation_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_result_path = str(Path(metrics_dir) / f"ablation_{model}_{exp_name}_{time_suffix}.json")
    conversation_result_path = str(Path(conversation_dir) / f"ablation_{model}_{exp_name}_{time_suffix}.json")

    # 设置环境变量
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    # 创建配置和环境
    from ReqElicitGym.config import ReqElicitGymConfig
    from ReqElicitGym.env import ReqElicitGym
    from TypoAgent.retriever import AblationInterviewer

    config = ReqElicitGymConfig(
        data_path=str(temp_data_path),
        judge_api_key=api_key,
        judge_base_url=base_url,
        judge_model_name=judge_model,
        judge_temperature=0.0,
        judge_max_tokens=1024,
        judge_timeout=timeout,
        user_api_key=api_key,
        user_base_url=base_url,
        user_model_name=user_model,
        user_temperature=0.0,
        user_max_tokens=1024,
        user_timeout=timeout,
        user_answer_quality="high",
        max_steps=max_steps,
        verbose=verbose,
        evaluation_result_path=evaluation_result_path,
        conversation_result_path=conversation_result_path,
    )

    # ReqElicitGymConfig 的 __post_init__ 会校验，需要确保 judge_api_key 和 user_api_key 不为 None
    # 上面已设置，直接 validate
    config.validate()

    print("=" * 60)
    print("消融实验配置")
    print("=" * 60)
    print(f"  消融组合: {exp_name}")
    print(f"    初始优先度: {use_initial_priority}")
    print(f"    上下文优先度: {use_context_priority}")
    print(f"    大类门控: {use_category_gating}")
    print(f"    大类确认阈值: cat_check={cat_check_threshold}, followup={followup_threshold}")
    print(f"  数据文件: {temp_data_path}")
    print(f"  树路径: {tree_path}")
    print(f"  模式: {mode} (任务数: {len(selected_tasks)})")
    print(f"  模型: {model}")
    print(f"  temperature: {temperature}")
    print(f"  对话: {conversation_result_path}")
    print(f"  指标: {evaluation_result_path}")
    print("=" * 60)

    env = ReqElicitGym(config)
    env.current_task_index = 0

    interviewer = AblationInterviewer(
        api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=2048,
        timeout=timeout,
        base_url=base_url,
        fixed_tree_path=tree_path,
        tree_percentage=tree_percentage,
        use_initial_priority=use_initial_priority,
        use_context_priority=use_context_priority,
        use_category_gating=use_category_gating,
        max_turns_per_category_no_gate=max_turns_per_category_no_gate if not use_category_gating else None,
        cat_check_threshold=cat_check_threshold,
        followup_threshold=followup_threshold,
    )

    print(f"Interviewer: {interviewer}\n")

    results = env.run_all_tasks(interviewer)

    try:
        env.save_evaluation_results(file_path=None, interviewer_model_name=interviewer.model_name)
        print(f"评估结果已保存: {evaluation_result_path}")
    except Exception as e:
        print(f"保存评估结果失败: {e}")
        import traceback
        traceback.print_exc()

    try:
        env.save_conversation_results(file_path=None)
        print(f"对话结果已保存: {conversation_result_path}")
    except Exception as e:
        print(f"保存对话结果失败: {e}")
        import traceback
        traceback.print_exc()

    if temp_data_path.exists():
        try:
            temp_data_path.unlink()
        except Exception:
            pass

    overall_metrics = results.get("overall_metrics", {})
    if overall_metrics:
        print("\n评估指标摘要:")
        print(f"  平均获取比例: {overall_metrics.get('elicitation_ratio', 0.0):.2%}")
        print(f"  平均 TKQR: {overall_metrics.get('tkqr', 0.0):.4f}")
        print(f"  平均 ORA: {overall_metrics.get('ora', 0.0):.4f}")

    print("\n完成.")
    return results


if __name__ == "__main__":
    main()
