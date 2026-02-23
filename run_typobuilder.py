#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypoBuilder 入口脚本：从 train.jsonl 采样得到 train_new.jsonl，构建 Typo_Tree 保存至 output/save_tree。
支持通过命令行参数或 main 内默认配置区域修改行为。
"""
import argparse
import sys
from pathlib import Path

# 将项目根目录加入 path，以便导入 TypoAgent
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TypoBuilder: 从 train.jsonl 采样并构建 Typo_Tree"
    )
    # ----- 采样相关 -----
    parser.add_argument(
        "--enable-sampling",
        action="store_true",
        help="启用采样：从 train.jsonl 采样得到 train_new.jsonl（默认关闭则直接使用已有 train_new.jsonl）",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["fixed", "proportional"],
        default="proportional",
        help="采样方式：fixed=每类固定k个，proportional=按比例采样",
    )
    parser.add_argument(
        "--sampling-k",
        type=int,
        default=5,
        help="fixed 模式下每类采样数",
    )
    parser.add_argument(
        "--sampling-total",
        type=int,
        default=168,
        help="proportional 模式下目标总量",
    )
    parser.add_argument(
        "--sampling-min-per-type",
        type=int,
        default=2,
        help="proportional 模式下每类最少采样数",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="采样随机种子",
    )
    # ----- 路径相关 -----
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 train.jsonl 路径（默认：TypoAgent/data/train.jsonl）",
    )
    parser.add_argument(
        "--sampled-output",
        type=str,
        default=None,
        help="采样后 train_new.jsonl 路径（默认：TypoAgent/data/train_new.jsonl）",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="树及中间文件保存目录（默认：output/save_tree）",
    )
    parser.add_argument(
        "--tree-output",
        type=str,
        default=None,
        help="最终 Typo_Tree 输出文件路径",
    )
    # ----- 构建树选项 -----
    parser.add_argument(
        "--use-layer2",
        action="store_true",
        help="使用 LLM 从 instruction 提炼 Layer2（默认关闭则从框架文件读取）",
    )
    parser.add_argument(
        "--use-layer3",
        action="store_true",
        default=True,
        help="使用骨架+instruction 抽取 Layer3（默认开启）",
    )
    parser.add_argument(
        "--no-use-layer3",
        action="store_false",
        dest="use_layer3",
        help="禁用 Layer3 抽取",
    )
    parser.add_argument(
        "--do-clustering",
        action="store_true",
        default=True,
        help="进行子类内聚类规整（默认开启）",
    )
    parser.add_argument(
        "--no-clustering",
        action="store_false",
        dest="do_clustering",
        help="禁用聚类",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API Key（也可用环境变量 OPENAI_API_KEY，或 DEFAULTS 中配置）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI API 基础 URL（也可用环境变量 OPENAI_BASE_URL，或 DEFAULTS 中配置）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM 模型名称（如 gpt-4o，也可在 DEFAULTS 中配置）",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ==================== 手动修改默认配置区域 ====================
    # 以下均可直接修改，覆盖命令行默认值（命令行传入的值仍会覆盖此处）
    import os
    DEFAULTS = {
        "enable_sampling": True,       # 是否启用采样
        "sampling_mode": "proportional",
        "sampling_k": 5,
        "sampling_total": 168,
        "sampling_min_per_type": 2,
        "sampling_seed": 42,
        "input_path": None,             # None 表示用 TypoAgent/data/train.jsonl
        "sampled_output_path": None,     # None 表示 TypoAgent/data/train_new.jsonl
        "save_dir": None,               # None 表示 output/save_tree
        "layer2_path": None,            # None 表示 {save_dir}/Typo_Tree_layer2.json
        "pre_cluster_path": None,       # None 表示 {save_dir}/Typo_Tree_incremental.json（增量构建输出，聚类输入）
        "tree_output": None,            # None 表示 {save_dir}/Typo_Tree.json（聚类输出）
        "use_layer2_extraction": True,
        "use_layer3_extraction": True,
        "do_clustering": True,
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),   # 可在代码中直接填入，None 使用 typo_builder 默认
        "model_name": "gpt-5.1",   # LLM 模型名称，如 gpt-4o、gpt-3.5-turbo 等
    }
    # ============================================================

    from TypoAgent.classification_sample import main as run_sampling
    from TypoAgent.typo_builder import build_typo_tree

    data_dir = _REPO_ROOT / "TypoAgent" / "data"
    save_tree_dir = _REPO_ROOT / "output" / "save_tree"

    input_path = args.input or DEFAULTS["input_path"] or str(data_dir / "train.jsonl")
    sampled_output_path = args.sampled_output or DEFAULTS["sampled_output_path"] or str(data_dir / "train_new.jsonl")
    save_dir = args.save_dir or DEFAULTS["save_dir"] or str(save_tree_dir)
    tree_output = args.tree_output or DEFAULTS["tree_output"] or str(Path(save_dir) / "Typo_Tree.json")

    enable_sampling = args.enable_sampling or DEFAULTS["enable_sampling"]
    use_layer2 = args.use_layer2 or DEFAULTS["use_layer2_extraction"]
    # use_layer3: 未传 CLI 时使用 DEFAULTS，传了 --use-layer3 或 --no-use-layer3 时以 CLI 为准
    use_layer3 = DEFAULTS["use_layer3_extraction"]
    if "--no-use-layer3" in sys.argv:
        use_layer3 = False
    elif "--use-layer3" in sys.argv:
        use_layer3 = True

    # 命令行覆盖
    if args.enable_sampling:
        enable_sampling = True
    if args.use_layer2:
        use_layer2 = True

    print("=" * 60)
    print("TypoBuilder 配置")
    print("=" * 60)
    print(f"  输入: {input_path}")
    print(f"  采样后: {sampled_output_path}")
    print(f"  保存目录: {save_dir}")
    _pre = DEFAULTS["pre_cluster_path"] or str(Path(save_dir) / "Typo_Tree_incremental.json")
    print(f"  增量构建输出: {_pre}")
    print(f"  最终输出: {tree_output}")
    print(f"  启用采样: {enable_sampling}")
    print(f"  Layer2 抽取: {use_layer2}")
    print(f"  Layer3 抽取: {use_layer3}")
    print(f"  聚类: {args.do_clustering}")
    model_name = args.model or DEFAULTS["model_name"]
    print(f"  LLM 模型: {model_name}")
    print("=" * 60)

    # 步骤 1：采样（可选）
    train_path_for_tree = input_path
    if enable_sampling:
        print("\n[1/2] 采样...")
        run_sampling(
            input_path=Path(input_path),
            output_path=Path(sampled_output_path),
            mode=args.sampling_mode,
            k=args.sampling_k,
            total=args.sampling_total,
            min_per_type=args.sampling_min_per_type,
            shuffle_output=True,
            seed=args.sampling_seed,
        )
        train_path_for_tree = sampled_output_path
    else:
        print("\n[跳过采样] 直接使用已有数据构建树")
        train_path_for_tree = sampled_output_path
        if not Path(train_path_for_tree).exists():
            print(f"警告: {train_path_for_tree} 不存在，将尝试使用 {input_path}")
            train_path_for_tree = input_path

    # 步骤 2：构建 Typo Tree
    print("\n[2/2] 构建 Typo Tree...")
    save_tree_path = Path(save_dir)
    save_tree_path.mkdir(parents=True, exist_ok=True)

    layer2_path = (DEFAULTS["layer2_path"] or str(save_tree_path / "Typo_Tree_layer2.json")) if use_layer2 else None
    pre_cluster_path = DEFAULTS["pre_cluster_path"] or str(save_tree_path / "Typo_Tree_incremental.json")
    framework_path = layer2_path if use_layer2 else str(data_dir / "LLMTree_struction.json")

    api_key = args.api_key or DEFAULTS["api_key"] or os.environ.get("OPENAI_API_KEY", "null")
    base_url = args.base_url or DEFAULTS["base_url"]

    if not api_key or api_key == "null":
        print("错误: 请设置 OPENAI_API_KEY 或 --api-key")
        sys.exit(1)

    build_typo_tree(
        train_path=train_path_for_tree,
        out_path=tree_output,
        framework_path=framework_path,
        layer2_output_path=layer2_path,
        pre_cluster_output_path=pre_cluster_path,
        use_layer2_extraction=use_layer2,
        use_layer3_extraction=use_layer3,
        do_clustering=args.do_clustering,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )

    print("\n完成。")


if __name__ == "__main__":
    main()
