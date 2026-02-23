#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 application_type 分层采样。支持两种采样方式：
- fixed: 每类固定采样 k 个
- proportional: 按比例采样，每类至少 2 个，总量约等于 total
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sample_fixed(groups: dict, k: int, shuffle: bool = False) -> list:
    """每类采样 k 个（不放回）。"""
    sampled_all = []
    for app_type, items in groups.items():
        n = min(k, len(items))
        sampled = random.sample(items, n)
        sampled_all.extend(sampled)
    if shuffle:
        random.shuffle(sampled_all)
    return sampled_all


def sample_proportional(
    groups: dict, total: int, min_per_type: int = 2, shuffle: bool = False
) -> list:
    """按 application_type 比例分层采样：数量多的类型多采，数量少的少采。"""
    total_in = sum(len(v) for v in groups.values())
    k_per_type = {}
    for app_type, items in groups.items():
        weight = len(items) / total_in if total_in > 0 else 0
        k_raw = max(min_per_type, round(total * weight))
        k_per_type[app_type] = min(k_raw, len(items))

    sampled_all = []
    for app_type, items in groups.items():
        n = k_per_type[app_type]
        sampled = random.sample(items, n)
        sampled_all.extend(sampled)

    if shuffle:
        random.shuffle(sampled_all)
    return sampled_all


def main(
    input_path: Path,
    output_path: Path,
    mode: str = "proportional",
    k: int = 5,
    total: int = 168,
    min_per_type: int = 2,
    shuffle_output: bool = False,
    seed: int = 42,
) -> int:
    """
    执行采样，返回采样后的数量。

    Args:
        input_path: 输入 jsonl 文件
        output_path: 输出 jsonl 文件
        mode: "fixed" 或 "proportional"
        k: fixed 模式下每类采样数
        total: proportional 模式下目标总量
        min_per_type: proportional 模式下每类至少采样数
        shuffle_output: 是否打乱输出顺序
        seed: 随机种子
    """
    random.seed(seed)

    groups = defaultdict(list)
    for obj in read_jsonl(input_path):
        app_type = obj.get("application_type", "")
        if not app_type:
            app_type = "__MISSING_APPLICATION_TYPE__"
        groups[app_type].append(obj)

    total_in = sum(len(v) for v in groups.values())

    if mode.lower() == "proportional":
        sampled_all = sample_proportional(
            groups, total=total, min_per_type=min_per_type, shuffle=shuffle_output
        )
        k_per_type = {}
        for app_type, items in groups.items():
            weight = len(items) / total_in if total_in > 0 else 0
            k_raw = max(min_per_type, round(total * weight))
            k_per_type[app_type] = min(k_raw, len(items))
    else:
        sampled_all = sample_fixed(groups, k=k, shuffle=shuffle_output)
        k_per_type = {t: min(k, len(v)) for t, v in groups.items()}

    write_jsonl(output_path, sampled_all)
    total_out = len(sampled_all)

    print(f"Loaded: {total_in} lines from {input_path}")
    print(f"Saved : {total_out} lines to   {output_path}")
    print(f"Mode: {mode} (k={k}, total={total}, min_per_type={min_per_type})")
    print("Per type sampled:")
    for t in sorted(k_per_type.keys()):
        print(f"  {t}: {k_per_type[t]}")

    return total_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Group by "application_type" and sample. Supports fixed (k per type) or proportional sampling.'
    )
    parser.add_argument("--input", default="train.jsonl", help="Input jsonl file")
    parser.add_argument("--output", default="train_new.jsonl", help="Output jsonl file")
    parser.add_argument(
        "--mode",
        choices=["fixed", "proportional"],
        default="proportional",
        help="fixed: k per type; proportional: weight by count, min 2 per type",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Sample size per application_type (fixed mode)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=168,
        help="Target total count (proportional mode)",
    )
    parser.add_argument(
        "--min-per-type",
        type=int,
        default=2,
        help="Min samples per type (proportional mode)",
    )
    parser.add_argument(
        "--shuffle-output",
        action="store_true",
        help="Shuffle final output across all types",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        input_path=Path(args.input),
        output_path=Path(args.output),
        mode=args.mode,
        k=args.k,
        total=args.total,
        min_per_type=args.min_per_type,
        shuffle_output=args.shuffle_output,
        seed=args.seed,
    )
