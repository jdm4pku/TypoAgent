"""Data structures for the static requirement tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TreeNode:
    """Tree node for the fixed requirement tree (leaf = question dimension)."""

    key: str
    question_seed: Optional[str] = None
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "TreeNode") -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path(self) -> str:
        parts: List[str] = []
        node: Optional[TreeNode] = self
        while node is not None:
            parts.append(node.key)
            node = node.parent
        return ".".join(reversed(parts))

    def iter_leaves(self) -> List["TreeNode"]:
        leaves: List[TreeNode] = []
        stack: List[TreeNode] = [self]
        while stack:
            cur = stack.pop()
            if cur.is_leaf():
                leaves.append(cur)
            else:
                for ch in reversed(cur.children):
                    stack.append(ch)
        return leaves
