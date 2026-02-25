
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TypoAgent 测试入口：读取树 -> 运行 ReqElicitGym 测试 -> 输出对话和指标。"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description="TypoAgent")
    parser.add_argument("--tree-path", type=str, default=None, help="树文件路径，默认 output/save_tree/LLMTree.auto-p.json")
    parser.add_argument("--mode", choices=["top3", "sample", "full", "test"], default=None, help="top3=前3个, sample=采样N个, full=全部, test=仅测top3的task0，不传则用 DEFAULTS 中的 mode")
    parser.add_argument("--sample-n", type=int, default=20, help="sample 模式采样数")
    parser.add_argument("--model", type=str, default=None, help="Interviewer 使用的 LLM 模型")
    parser.add_argument("--temperature", type=float, default=None, help="Interviewer LLM 的 temperature")
    parser.add_argument("--api-key", type=str, default=None, help="API Key，也可用 OPENAI_API_KEY")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge 模型")
    parser.add_argument("--user-model", type=str, default=None, help="User Simulator 模型")
    parser.add_argument("--output-conversation-dir", type=str, default=None, help="对话输出目录")
    parser.add_argument("--output-metrics-dir", type=str, default=None, help="指标输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--cat-check-threshold", type=int, default=None, help="大类确认阈值：连续多少轮无新增后触发第一次大类确认，默认2")
    parser.add_argument("--followup-threshold", type=int, default=None, help="大类追问阈值：第一次确认后连续多少轮无新增再触发追问，默认3")
    return parser


def main():
    args = build_parser().parse_args()
    # ==================== 手动修改默认配置区域 ====================
    DEFAULTS = {
        "tree_path": "/home/ubuntu/jdm/xiaotian/TypoAgent_release_v2/output/save_tree/Typo_Tree.json", "mode": "test", "sample_n": 20, "model": "gpt-5.1",
        "temperature": 0.0,
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.chatanywhere.tech/v1",
        "judge_model": "gpt-5.1", "user_model": "gpt-5.1",
        "output_conversation_dir": None, "output_metrics_dir": None, "verbose": True,
        "cat_check_threshold": 2,
        "followup_threshold": 2,
    }
    # ============================================================
    tree_path = args.tree_path or DEFAULTS["tree_path"] or str(_REPO_ROOT / "output" / "save_tree" / "LLMTree.auto-p.json")
    mode = args.mode or DEFAULTS["mode"]
    sample_n = args.sample_n or DEFAULTS["sample_n"]
    model = args.model or DEFAULTS["model"]
    temperature = args.temperature if args.temperature is not None else DEFAULTS["temperature"]
    api_key = args.api_key or DEFAULTS["api_key"]
    base_url = args.base_url or DEFAULTS["base_url"]
    judge_model = args.judge_model or DEFAULTS["judge_model"]
    user_model = args.user_model or DEFAULTS["user_model"]
    conversation_dir = args.output_conversation_dir or DEFAULTS["output_conversation_dir"] or str(_REPO_ROOT / "output" / "conversation")
    metrics_dir = args.output_metrics_dir or DEFAULTS["output_metrics_dir"] or str(_REPO_ROOT / "output" / "metrics")
    verbose = args.verbose or DEFAULTS["verbose"]
    cat_check_threshold = args.cat_check_threshold if args.cat_check_threshold is not None else DEFAULTS["cat_check_threshold"]
    followup_threshold = args.followup_threshold if args.followup_threshold is not None else DEFAULTS["followup_threshold"]
    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 或 --api-key"); sys.exit(1)
    data_dir = _REPO_ROOT / "ReqElicitGym" / "data"
    if mode == "top3":
        top3_path = data_dir / "test_top3.json"
        if top3_path.exists():
            data_path = str(top3_path)
            selected = json.loads(top3_path.read_text(encoding="utf-8"))
            temp_data_path = None
        else:
            source_data = data_dir / "test.json"
            if not source_data.exists():
                print("错误: 找不到 test.json 或 test_top3.json"); sys.exit(1)
            with open(source_data, "r", encoding="utf-8") as f:
                all_tasks = json.load(f)
            selected = all_tasks[:3]
            temp_data_path = data_dir / "test_typoagent_top3.json"
            temp_data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_data_path, "w", encoding="utf-8") as f:
                json.dump(selected, f, ensure_ascii=False, indent=2)
            data_path = str(temp_data_path)
    elif mode == "test":
        top3_path = data_dir / "test_top3.json"
        if top3_path.exists():
            all_top3 = json.loads(top3_path.read_text(encoding="utf-8"))
            selected = all_top3[:1] if all_top3 else []
        else:
            source_data = data_dir / "test.json"
            if not source_data.exists():
                print("错误: 找不到 test.json 或 test_top3.json"); sys.exit(1)
            with open(source_data, "r", encoding="utf-8") as f:
                all_tasks = json.load(f)
            selected = all_tasks[:1]
        if not selected:
            print("错误: top3 中无 task0"); sys.exit(1)
        temp_data_path = data_dir / "test_typoagent_task0.json"
        temp_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_data_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        data_path = str(temp_data_path)
    elif mode == "sample":
        source_data = data_dir / "test.json"
        if not source_data.exists():
            print("错误: 找不到 test.json"); sys.exit(1)
        with open(source_data, "r", encoding="utf-8") as f:
            all_tasks = json.load(f)
        import random
        selected = random.sample(all_tasks, min(sample_n, len(all_tasks)))
        temp_data_path = data_dir / f"test_typoagent_{mode}.json"
        temp_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_data_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        data_path = str(temp_data_path)
    else:
        source_data = data_dir / "test.json"
        if not source_data.exists():
            print("错误: 找不到 test.json"); sys.exit(1)
        with open(source_data, "r", encoding="utf-8") as f:
            selected = json.load(f)
        data_path = str(source_data)
        temp_data_path = None
    os.makedirs(conversation_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_result_path = str(Path(metrics_dir) / f"typoagent_{model}_{mode}_{time_suffix}.json")
    conversation_result_path = str(Path(conversation_dir) / f"typoagent_{model}_{mode}_{time_suffix}.json")
    from ReqElicitGym.config import ReqElicitGymConfig
    from ReqElicitGym.env import ReqElicitGym
    from TypoAgent.retriever import TypoAgentInterviewer
    config = ReqElicitGymConfig(
        data_path=data_path,
        judge_api_key=api_key, judge_base_url=base_url, judge_model_name=judge_model,
        judge_temperature=0.0, judge_max_tokens=1024, judge_timeout=30.0,
        user_api_key=api_key, user_base_url=base_url, user_model_name=user_model,
        user_temperature=0.0, user_max_tokens=1024, user_timeout=30.0,
        user_answer_quality="high", max_steps=20, verbose=verbose,
        evaluation_result_path=evaluation_result_path,
        conversation_result_path=conversation_result_path,
    )
    config.validate()
    print("=" * 60)
    print("TypoAgent 配置")
    print("=" * 60)
    print(f"  数据文件: {data_path}")
    print(f"  树路径: {tree_path}")
    print(f"  模式: {mode} (任务数: {len(selected)})")
    print(f"  模型: {model}")
    print(f"  temperature: {temperature}")
    print(f"  对话: {conversation_result_path}")
    print(f"  指标: {evaluation_result_path}")
    print(f"  大类确认阈值: cat_check={cat_check_threshold}, followup={followup_threshold}")
    print("=" * 60)
    env = ReqElicitGym(config)
    env.current_task_index = 0  # __init__ 中 reset() 已消耗 task_0，需重置以便 run_all_tasks 从 task_0 开始
    interviewer = TypoAgentInterviewer(api_key=api_key, model_name=model, temperature=temperature, max_tokens=2048, timeout=30.0, base_url=base_url, fixed_tree_path=tree_path, tree_percentage=100.0, cat_check_threshold=cat_check_threshold, followup_threshold=followup_threshold)
    print(f"Interviewer: {interviewer}\n")
    results = env.run_all_tasks(interviewer)
    try:
        env.save_evaluation_results(file_path=None, interviewer_model_name=interviewer.model_name)
    except Exception as e:
        print(f"保存评估结果时出错: {e}")
    try:
        env.save_conversation_results(file_path=None)
    except Exception as e:
        print(f"保存对话时出错: {e}")
    try:
        if temp_data_path is not None and temp_data_path.exists():
            temp_data_path.unlink()
    except Exception:
        pass
    overall = results.get("overall_metrics", {})
    if overall:
        print("\n评估指标:")
        print(f"  总任务数: {overall.get('total_tasks', 0)}")
        print(f"  平均获取比例: {overall.get('elicitation_ratio', 0):.2%}")
        print(f"  平均 TKQR: {overall.get('tkqr', 0):.4f}")
        print(f"  平均 ORA: {overall.get('ora', 0):.4f}")
    print("\n完成.")


if __name__ == "__main__":
    main()
