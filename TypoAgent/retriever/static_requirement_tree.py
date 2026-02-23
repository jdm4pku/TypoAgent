"""Static requirement tree: fixed order only.

- Load tree from JSON once; build leaf_catalog and leaf_order.
- next_leaf() returns the first leaf in leaf_order that should_ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .fixed_tree_loader import load_fixed_tree
from .schema import TreeNode


CATEGORY_ORDER = ["Interaction", "Content", "Style"]

PARENT_QUESTION_TEMPLATE = (
    "For the dimension «{sub_key}» (covering aspects such as {aspects}), "
    "could you tell me your specific preferences? Please describe what you have in mind, "
    "for example: which options you prefer, or any concrete requirements."
)


@dataclass
class StaticRequirementTree:
    fixed_tree_path: str
    max_ask_per_leaf: int = 2
    percentage: float = 100.0

    root: TreeNode = field(init=False)
    leaf_catalog: Dict[str, str] = field(init=False)
    leaf_order: List[str] = field(init=False)
    parent_leaf_ids: Set[str] = field(init=False)

    asked_count: Dict[str, int] = field(default_factory=dict)
    covered: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # leaf_id -> (value, status)

    def __post_init__(self) -> None:
        tree_path = Path(self.fixed_tree_path)
        if not tree_path.exists():
            # 尝试 output/save_tree 等路径
            repo_root = Path(__file__).resolve().parents[2]
            for sub in ["output/save_tree", ""]:
                candidate = repo_root / sub / Path(self.fixed_tree_path).name
                if candidate.exists():
                    tree_path = candidate
                    break
        self.fixed_tree_path = str(tree_path)

        self.root = load_fixed_tree(self.fixed_tree_path, percentage=self.percentage, export_pruned=False)
        self.leaf_catalog, self.parent_leaf_ids = self._build_leaf_catalog_with_parents(self.root)
        self.leaf_order = self._build_leaf_order(self.root)
        self.asked_count = {lid: 0 for lid in self.leaf_catalog}

    def reset_state(self) -> None:
        self.asked_count = {lid: 0 for lid in self.leaf_catalog}
        self.covered = {}

    def mark_covered(self, leaf_id: str, value: str, status: str) -> None:
        if leaf_id in self.leaf_catalog and status in ("need", "no_need"):
            self.covered[leaf_id] = (value, status)

    def mark_asked(self, leaf_id: str) -> None:
        if leaf_id in self.asked_count:
            self.asked_count[leaf_id] += 1

    def is_covered(self, leaf_id: str) -> bool:
        return leaf_id in self.covered

    def max_ask_for_leaf(self, leaf_id: str) -> int:
        seed = (self.leaf_catalog.get(leaf_id) or "").strip().lower()
        if not seed:
            return self.max_ask_per_leaf
        yn = ("do you", "should", "is there", "are there", "would you", "can you", "need ")
        if seed.startswith(yn) or seed.endswith("?"):
            return 1 if self.max_ask_per_leaf >= 1 else 0
        return self.max_ask_per_leaf

    def should_ask(self, leaf_id: str) -> bool:
        if leaf_id not in self.leaf_catalog:
            return False
        if self.is_covered(leaf_id):
            return False
        return self.asked_count.get(leaf_id, 0) < self.max_ask_for_leaf(leaf_id)

    def get_seed_question(self, leaf_id: str) -> str:
        return self.leaf_catalog.get(leaf_id, "")

    def next_leaf(self) -> Optional[str]:
        for leaf_id in self.leaf_order:
            if self.should_ask(leaf_id):
                return leaf_id
        return None

    def next_leaf_from_categories(self, categories: Set[str]) -> Optional[str]:
        for leaf_id in self.leaf_order:
            top = leaf_id.split(".", 1)[0]
            if top in categories and self.should_ask(leaf_id):
                return leaf_id
        return None

    def all_determined(self) -> bool:
        for leaf_id in self.leaf_catalog:
            if not self.is_covered(leaf_id) and self.asked_count.get(leaf_id, 0) < self.max_ask_for_leaf(leaf_id):
                return False
        return True

    def _build_leaf_catalog_with_parents(self, root: TreeNode) -> Tuple[Dict[str, str], Set[str]]:
        catalog: Dict[str, str] = {}
        parent_ids: Set[str] = set()
        for cat in root.children:
            cat_key = cat.key
            for sub in cat.children:
                if not sub.is_leaf():
                    parent_id = f"{cat_key}.{sub.key}"
                    parent_ids.add(parent_id)
                    catalog[parent_id] = self._parent_seed_question(sub)
                for leaf in sub.iter_leaves():
                    if leaf.question_seed:
                        path = leaf.path().replace("ROOT.", "", 1)
                        catalog[path] = leaf.question_seed
        return catalog, parent_ids

    def _parent_seed_question(self, sub_node: TreeNode) -> str:
        leaves = sub_node.iter_leaves()
        if not leaves:
            return PARENT_QUESTION_TEMPLATE.format(sub_key=sub_node.key, aspects="various sub-dimensions")
        parts = [leaf.key.replace("_", " ") for leaf in leaves[:3]]
        if len(leaves) > 3:
            aspects = ", ".join(parts) + ", and related aspects"
        else:
            aspects = ", ".join(parts)
        return PARENT_QUESTION_TEMPLATE.format(sub_key=sub_node.key, aspects=aspects)

    def child_leaf_ids(self, parent_id: str) -> List[str]:
        prefix = parent_id + "."
        return [lid for lid in self.leaf_catalog if lid.startswith(prefix)]

    def mark_subtree_no_need(self, parent_id: str, reason: str = "parent no_need") -> None:
        for child_id in self.child_leaf_ids(parent_id):
            if not self.is_covered(child_id):
                self.mark_covered(child_id, reason, "no_need")

    def mark_category_no_need(self, category: str, reason: str = "category no_need") -> None:
        for leaf_id in self.leaf_catalog.keys():
            top = leaf_id.split(".", 1)[0]
            if top == category and not self.is_covered(leaf_id):
                self.mark_covered(leaf_id, reason, "no_need")

    def _build_leaf_order(self, root: TreeNode) -> List[str]:
        cats = {c.key: c for c in root.children}
        ordered_cats: List[TreeNode] = []
        for k in CATEGORY_ORDER:
            if k in cats:
                ordered_cats.append(cats[k])
        for c in root.children:
            if c.key not in CATEGORY_ORDER:
                ordered_cats.append(c)

        leaf_order: List[str] = []
        for cat in ordered_cats:
            for sub in cat.children:
                parent_id = f"{cat.key}.{sub.key}"
                if parent_id in self.leaf_catalog:
                    leaf_order.append(parent_id)
                for leaf in sub.iter_leaves():
                    if leaf.question_seed:
                        leaf_order.append(leaf.path().replace("ROOT.", "", 1))
        return leaf_order
