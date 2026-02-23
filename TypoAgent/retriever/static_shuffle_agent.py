"""StaticTree_shuffle_priority: 新 episode 时根据初始需求对 Layer2 静态重排。"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .prompt_loader import get_prompt
from .requirement_tree_agent import RequirementTreeAgent


class StaticShuffleAgent(RequirementTreeAgent):
    """在 base 基础上增加：按 LLM 打分静态重排 leaf 顺序。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_state_last_initial = None

    def reset(self):
        super().reset()
        self._init_state_last_initial = None

    def _maybe_reset_episode(self, conversation_history):
        if not conversation_history:
            self.reset()
            return
        if len(conversation_history) == 1 and conversation_history[0].get("role") == "user":
            initial = (conversation_history[0].get("content") or "").strip()
            if self._init_state_last_initial != initial:
                self.reset()
                self._init_state_last_initial = initial
                self._reorder_leaf_order_by_initial(initial)
            return
        if len(conversation_history) < self.state.last_processed_len:
            self.reset()

    def _layer2_id(self, lid):
        parts = lid.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else lid

    def _score_layer2_in_category(self, initial, category, layer2_units):
        if not layer2_units:
            return {}
        units_block_parts = []
        for layer2_id, infos in layer2_units.items():
            desc = [f"  layer2_id={layer2_id}"]
            for info in infos:
                desc.append(f"    - leaf_id={info['leaf_id']}, seed_question={info['seed_question']}")
            units_block_parts.append("\n".join(desc))
        units_block = "\n\n".join(units_block_parts)
        sys_p = get_prompt("InitialPriorityScoring_system").format(category=category)
        usr_p = get_prompt("InitialPriorityScoring_user").format(initial=initial, category=category, units_block=units_block)
        try:
            resp = self.model_call(sys_p, usr_p, self.model_config, return_json=True)
        except Exception:
            return {}
        scores = {}
        if isinstance(resp, dict):
            raw = resp.get("scores") or resp.get("layer2_scores") or []
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    lid = item.get("layer2_id")
                    if not isinstance(lid, str) or lid not in layer2_units:
                        continue
                    try:
                        scores[lid] = float(item.get("score", 0))
                    except (TypeError, ValueError):
                        continue
        return scores

    def _reorder_leaf_order_by_initial(self, initial):
        initial = (initial or "").strip()
        if not initial:
            return
        catalog = self.tree.leaf_catalog
        leaf_order = list(self.tree.leaf_order)
        if not leaf_order:
            return
        by_cat = {}
        for lid in leaf_order:
            top = lid.split(".", 1)[0]
            by_cat.setdefault(top, []).append(lid)
        category_order = []
        for cat in ("Interaction", "Content", "Style"):
            if cat in by_cat:
                category_order.append(cat)
        for cat in sorted(by_cat.keys()):
            if cat not in category_order:
                category_order.append(cat)
        new_order = []
        for cat in category_order:
            items = by_cat[cat]
            if not items:
                continue
            by_layer2 = {}
            for lid in items:
                l2 = self._layer2_id(lid)
                by_layer2.setdefault(l2, []).append(lid)
            layer2_units = {}
            for layer2_id, lids in by_layer2.items():
                infos = [{"leaf_id": lid, "seed_question": (catalog.get(lid, "") or "").strip()} for lid in lids]
                layer2_units[layer2_id] = infos
            layer2_scores = self._score_layer2_in_category(initial, cat, layer2_units)
            if layer2_scores:
                layer2_order = sorted(
                    by_layer2.keys(),
                    key=lambda l2: (-layer2_scores.get(l2, 0.0), min(leaf_order.index(lid) for lid in by_layer2[l2])),
                )
                for l2 in layer2_order:
                    new_order.extend(by_layer2[l2])
            else:
                new_order.extend(items)
        if new_order:
            self.tree.leaf_order = new_order
