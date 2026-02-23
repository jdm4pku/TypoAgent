"""Load the fixed requirement tree from JSON."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .schema import TreeNode


def _prune_tree_data_by_percentage(data: Dict[str, Any], percentage: float) -> Dict[str, Any]:
    if percentage >= 100:
        return data
    rng = random.Random(42)
    keep_ratio = max(0.0, min(percentage, 100.0)) / 100.0
    new_data: Dict[str, Any] = {}
    for cat_key, cat_val in data.items():
        if not isinstance(cat_val, dict):
            new_data[cat_key] = cat_val
            continue
        leaves: List[Tuple[str, str]] = []
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, dict):
                continue
            for leaf_key in sub_val.keys():
                leaves.append((sub_key, leaf_key))
        total = len(leaves)
        if total == 0:
            new_data[cat_key] = cat_val
            continue
        keep_count = int(round(total * keep_ratio))
        if keep_count <= 0:
            keep_count = 1
        if keep_count >= total:
            new_data[cat_key] = cat_val
            continue
        kept = set(rng.sample(leaves, keep_count))
        pruned_cat: Dict[str, Any] = {}
        for sub_key, sub_val in cat_val.items():
            if not isinstance(sub_val, dict):
                pruned_cat[sub_key] = sub_val
                continue
            new_sub: Dict[str, Any] = {}
            for leaf_key, leaf_q in sub_val.items():
                if (sub_key, leaf_key) in kept:
                    new_sub[leaf_key] = leaf_q
            if new_sub:
                pruned_cat[sub_key] = new_sub
        new_data[cat_key] = pruned_cat
    return new_data


def load_fixed_tree(
    json_path: str,
    percentage: float = 100.0,
    export_pruned: bool = False,
    export_suffix: Optional[str] = None,
) -> TreeNode:
    path = Path(json_path)
    if not path.is_absolute() and not path.exists():
        # 尝试相对于常见根目录
        candidates = [
            Path(__file__).resolve().parents[2] / "output" / "save_tree" / path.name,
            Path(__file__).resolve().parents[2] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    raw_data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if percentage < 100.0:
        data = _prune_tree_data_by_percentage(raw_data, percentage)
        if export_pruned:
            ep = path.with_name(f"{path.stem}_{export_suffix or int(percentage)}{path.suffix}")
            try:
                ep.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
    else:
        data = raw_data

    root = TreeNode("ROOT")
    for cat_key, cat_val in data.items():
        cat_node = TreeNode(cat_key)
        root.add_child(cat_node)
        if not isinstance(cat_val, dict):
            continue
        for sub_key, sub_val in cat_val.items():
            sub_node = TreeNode(sub_key)
            cat_node.add_child(sub_node)
            if not isinstance(sub_val, dict):
                continue
            for leaf_key, leaf_q in sub_val.items():
                leaf_node = TreeNode(leaf_key, question_seed=str(leaf_q))
                sub_node.add_child(leaf_node)
    return root
