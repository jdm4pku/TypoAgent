#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comp3 测评入口：使用 Requirements-Elicitation-Follow-Up-Question-Generation 方法进行 ReqElicitGym 测评。

参考: https://github.com/anmolsinghal98/Requirements-Elicitation-Follow-Up-Question-Generation
论文: Y. Shen, A. Singhal, T.D. Breaux (2025). "Requirements Elicitation Follow-up Question Generation."
      33rd IEEE International Requirements Engineering Conference.

运行后使用 ReqElicitGym 的测评规则输出：
- elicitation_ratio: 需求获取比例
- tkqr: Turn-discounted Key Question Rate
- ora: Optimal Round Assessment
- action_type_effectiveness: 动作类型有效性
- aspect_type_elicitation: 方面类型获取比例
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Comp3: Requirements-Elicitation-Follow-Up-Question-Generation 测评"
    )
    parser.add_argument(
        "--mode",
        choices=["top3", "sample", "full"],
        default=None,
        help="top3=前3个任务, sample=采样N个, full=全部，不传则用 DEFAULTS",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="sample 模式采样数，不传则用 DEFAULTS",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Interviewer 使用的 LLM 模型（默认 gpt-4o，与论文一致）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Interviewer LLM 的 temperature",
    )
    parser.add_argument(
        "--finish",
        action="store_true",
        help="启用 finish 动作（允许模型输出 user stories 结束对话）；默认不启用，跑到 max_steps 截断",
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
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # ==================== 默认配置 ====================
    DEFAULTS = {
        "mode": "full",
        "sample_n": 20,
        "model": "gpt-5.1",
        "temperature": 0.0,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.chatanywhere.tech/v1",
        "judge_model": "gpt-5.1",
        "user_model": "gpt-5.1",
        "output_conversation_dir": None,
        "output_metrics_dir": None,
        "verbose": True,
    }
    # ============================================================

    mode = args.mode or DEFAULTS["mode"]
    sample_n = args.sample_n if args.sample_n is not None else DEFAULTS["sample_n"]
    model = args.model or DEFAULTS["model"]
    temperature = (
        args.temperature if args.temperature is not None else DEFAULTS["temperature"]
    )
    api_key = args.api_key or DEFAULTS["api_key"]
    base_url = args.base_url or DEFAULTS["base_url"]
    judge_model = args.judge_model or DEFAULTS["judge_model"]
    user_model = args.user_model or DEFAULTS["user_model"]
    conversation_dir = (
        args.output_conversation_dir
        or DEFAULTS["output_conversation_dir"]
        or str(_REPO_ROOT / "output" / "conversation")
    )
    metrics_dir = (
        args.output_metrics_dir
        or DEFAULTS["output_metrics_dir"]
        or str(_REPO_ROOT / "output" / "metrics")
    )
    verbose = args.verbose or DEFAULTS["verbose"]
    enable_finish = args.finish

    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 或 --api-key")
        sys.exit(1)

    data_dir = _REPO_ROOT / "ReqElicitGym" / "data"
    source_data = data_dir / "test.json"
    if not source_data.exists():
        print("错误: 找不到 ReqElicitGym/data/test.json")
        sys.exit(1)

    temp_data_path = None
    with open(source_data, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)

    if mode == "top3":
        top3_path = data_dir / "test_top3.json"
        if top3_path.exists():
            data_path = str(top3_path)
            selected = json.loads(top3_path.read_text(encoding="utf-8"))
        else:
            selected = all_tasks[:3]
            temp_dir = _REPO_ROOT / "output" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_data_path = temp_dir / "test_comp3_top3.json"
            with open(temp_data_path, "w", encoding="utf-8") as f:
                json.dump(selected, f, ensure_ascii=False, indent=2)
            data_path = str(temp_data_path)
    elif mode == "sample":
        selected = random.sample(all_tasks, min(sample_n, len(all_tasks)))
        temp_dir = _REPO_ROOT / "output" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_data_path = temp_dir / "test_comp3_sample.json"
        with open(temp_data_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        data_path = str(temp_data_path)
    else:  # full
        selected = all_tasks
        data_path = str(source_data)

    os.makedirs(conversation_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_result_path = str(
        Path(metrics_dir) / f"comp3_{model}_{mode}_{time_suffix}.json"
    )
    conversation_result_path = str(
        Path(conversation_dir) / f"comp3_{model}_{mode}_{time_suffix}.json"
    )

    from ReqElicitGym.config import ReqElicitGymConfig
    from ReqElicitGym.env import ReqElicitGym
    from comp3.comp3_interviewer import Comp3Interviewer

    config = ReqElicitGymConfig(
        data_path=data_path,
        judge_api_key=api_key,
        judge_base_url=base_url,
        judge_model_name=judge_model,
        judge_temperature=0.0,
        judge_max_tokens=1024,
        judge_timeout=60.0,
        user_api_key=api_key,
        user_base_url=base_url,
        user_model_name=user_model,
        user_temperature=0.0,
        user_max_tokens=1024,
        user_timeout=60.0,
        user_answer_quality="high",
        max_steps=20,
        verbose=verbose,
        evaluation_result_path=evaluation_result_path,
        conversation_result_path=conversation_result_path,
    )
    config.validate()

    from pathlib import Path as _P
    _log_dir = _P.cwd() / "output" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _diag_log = _log_dir / "reqelicit_diagnostic.log"

    print("=" * 60)
    print("Comp3 (Requirements-Elicitation-Follow-Up) 测评配置")
    print("=" * 60)
    print(f"  诊断日志: {_diag_log}")
    print(f"  数据文件: {data_path}")
    print(f"  模式: {mode} (任务数: {len(selected)})")
    print(f"  模型: {model}")
    print(f"  temperature: {temperature}")
    print(f"  enable_finish: {enable_finish} (False=仅跟进问题，跑到 max_steps=20 截断)")
    print(f"  base_url: {base_url or '(default)'}")
    print(f"  对话输出: {conversation_result_path}")
    print(f"  指标输出: {evaluation_result_path}")
    print("=" * 60)

    env = ReqElicitGym(config)
    env.current_task_index = 0
    interviewer = Comp3Interviewer(
        api_key=api_key,
        model_name=model,
        temperature=temperature,
        max_tokens=1024,
        timeout=60.0,
        base_url=base_url,
        enable_finish=enable_finish,
    )
    print(f"Interviewer: {interviewer}\n")

    results = env.run_all_tasks(interviewer)

    try:
        env.save_evaluation_results(
            file_path=None, interviewer_model_name=interviewer.model_name
        )
        print(f"\n评估结果已保存到: {evaluation_result_path}")
    except Exception as e:
        print(f"保存评估结果时出错: {e}")

    try:
        env.save_conversation_results(file_path=None)
        print(f"对话过程已保存到: {conversation_result_path}")
    except Exception as e:
        print(f"保存对话时出错: {e}")

    try:
        if temp_data_path is not None and temp_data_path.exists():
            temp_data_path.unlink()
            print(f"\n已清理临时文件: {temp_data_path}")
    except Exception:
        pass

    overall = results.get("overall_metrics", {})
    if overall:
        print("\n" + "=" * 60)
        print("ReqElicitGym 测评结果")
        print("=" * 60)
        print(f"  总测试样本数: {overall.get('total_tasks', 0)}")
        print(f"  总隐式需求数: {overall.get('total_requirements_all_tasks', 0)}")
        print(f"  总获取数: {overall.get('total_elicited_all_tasks', 0)}")
        print("\n平均指标（基于测试样本平均）:")
        print(f"  平均获取比例 (elicitation_ratio): {overall.get('elicitation_ratio', 0):.2%}")
        print(f"  平均 TKQR: {overall.get('tkqr', 0):.4f}")
        print(f"  平均 ORA: {overall.get('ora', 0):.4f}")
        app_type_stats = overall.get("application_type_statistics", {})
        if app_type_stats:
            print("\n按应用类型统计:")
            print(
                f"{'Application Type':<40} {'任务数':<10} {'平均获取比例':<15} {'平均TKQR':<12} {'平均ORA':<12}"
            )
            print("-" * 100)
            for app_type in sorted(app_type_stats.keys()):
                stats = app_type_stats[app_type]
                print(
                    f"{app_type:<40} {stats['num_tasks']:<10} "
                    f"{stats['average_elicitation_ratio']:>13.2%} "
                    f"{stats['average_tkqr']:>10.4f} "
                    f"{stats['average_ora']:>10.4f}"
                )
        if overall.get("action_type_effectiveness"):
            print("\n动作类型有效性:")
            for action_type, stats in overall["action_type_effectiveness"].items():
                print(
                    f"  {action_type}: {stats['effective']}/{stats['total']} = {stats['effectiveness_ratio']:.2%}"
                )
        if overall.get("aspect_type_elicitation"):
            print("\n方面类型获取比例:")
            for aspect, stats in overall["aspect_type_elicitation"].items():
                if stats["total"] > 0:
                    print(
                        f"  {aspect}: {stats['elicited']}/{stats['total']} = {stats['elicitation_ratio']:.2%}"
                    )
        print("\n完成.")


if __name__ == "__main__":
    main()
