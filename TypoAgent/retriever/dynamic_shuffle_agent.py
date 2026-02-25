"""DynamicTree_shuffle_priority: 每次提问前根据上下文动态重排 + 大类级要/不要确认。"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set

from .implicit_topic_parser import parse_user_message_to_leaves
from .prompt_loader import get_prompt
from .static_shuffle_agent import StaticShuffleAgent


class DynamicShuffleAgent(StaticShuffleAgent):
    """在 StaticShuffle 基础上增加：每次提问前按上下文动态重排 + 大类级确认。"""

    def __init__(self, *args, use_judge_for_need: bool = True, cat_check_threshold: int = 2, followup_threshold: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_judge_for_need = use_judge_for_need
        self._cat_check_threshold = cat_check_threshold
        self._followup_threshold = followup_threshold
        self._last_was_category_check = False
        self._pending_category_check_category = None
        self._category_check_asked = set()
        self._categories_asked = set()
        self._style_deferred_count = 0
        self._max_style_deferred = 2
        self._expanded_layer2 = set()
        self._category_followup_asked = set()

    def reset(self):
        super().reset()
        self._last_was_category_check = False
        self._pending_category_check_category = None
        self._category_check_asked = set()
        self._categories_asked = set()
        self._style_deferred_count = 0
        self._expanded_layer2 = set()
        self._category_followup_asked = set()

    def ask_question(self, conversation_history):
        self._maybe_reset_episode(conversation_history)
        self._update_state_from_history(conversation_history)
        # 先判断结束和大类确认，避免不必要的上下文重排 LLM 调用
        if self.tree.all_determined() and len(self._categories_asked) >= 3:
            return self._finish(conversation_history)
        if self.state.turns_asked >= self.max_turns:
            return self._finish(conversation_history)
        current_cat = getattr(self, "_current_category", None)
        cat_check_threshold = getattr(self, "_cat_check_threshold", 2)
        followup_threshold = getattr(self, "_followup_threshold", 3)
        first_check = (
            current_cat
            and current_cat not in getattr(self, "_categories_done", set())
            and self._no_new_in_cat_streak >= cat_check_threshold
            and current_cat not in self._category_check_asked
        )
        followup_check = (
            current_cat
            and current_cat not in getattr(self, "_categories_done", set())
            and self._no_new_in_cat_streak >= followup_threshold
            and current_cat in self._category_check_asked
            and current_cat not in self._category_followup_asked
        )
        if first_check or followup_check:
            if current_cat == "Style" and self._style_deferred_count < self._max_style_deferred:
                cat_leaf = self.tree.next_leaf_from_categories({"Style"})
                if cat_leaf is not None:
                    self._style_deferred_count += 1
                    seed = self.tree.get_seed_question(cat_leaf)
                    question = self._ask_leaf(conversation_history, cat_leaf, seed)
                    if question:
                        self.tree.mark_asked(cat_leaf)
                        self._last_asked_leaf_id = cat_leaf
                        self._categories_asked.add("Style")
                        self.state.turns_asked += 1
                        return question
                    self.tree.mark_asked(cat_leaf)
            self._last_was_category_check = True
            self._pending_category_check_category = current_cat
            if followup_check:
                self._category_followup_asked.add(current_cat)
            else:
                self._category_check_asked.add(current_cat)
            self.state.turns_asked += 1
            return self._format_category_check_question(current_cat, is_followup=followup_check)
        # 需要选下一个 leaf 时，再按上下文动态重排
        self._reorder_leaf_order_by_context(conversation_history)
        leaf_id = self.tree.next_leaf()
        if leaf_id is None and len(self._categories_asked) < 3:
            missing = {"Interaction", "Content", "Style"} - self._categories_asked
            if missing:
                leaf_id = self.tree.next_leaf_from_categories(missing)
        if leaf_id is None:
            return self._finish(conversation_history)
        top_cat = leaf_id.split(".", 1)[0]
        if self._current_category != top_cat:
            self._current_category = top_cat
            self._no_new_in_cat_streak = 0
        seed = self.tree.get_seed_question(leaf_id)
        question = self._ask_leaf(conversation_history, leaf_id, seed)
        if not question:
            self.tree.mark_asked(leaf_id)
            self._categories_asked.add(leaf_id.split(".", 1)[0])
            leaf_id2 = self.tree.next_leaf()
            if leaf_id2 is None:
                return self._finish(conversation_history)
            seed2 = self.tree.get_seed_question(leaf_id2)
            question = self._ask_leaf(conversation_history, leaf_id2, seed2)
            self.tree.mark_asked(leaf_id2)
            self._last_asked_leaf_id = leaf_id2
            self._categories_asked.add(leaf_id2.split(".", 1)[0])
            self.state.turns_asked += 1
            return question or seed2
        self.tree.mark_asked(leaf_id)
        self._last_asked_leaf_id = leaf_id
        self._categories_asked.add(top_cat)
        self.state.turns_asked += 1
        return question

    def _update_state_from_history(self, conversation_history):
        start = max(self.state.last_processed_len, 0)
        new_msgs = conversation_history[start:]
        for msg in new_msgs:
            if msg.get("role") != "user":
                continue
            user_text = (msg.get("content") or "").strip()
            if not user_text:
                continue
            if self._last_was_category_check and self._pending_category_check_category is not None:
                cat = self._pending_category_check_category
                decision = self._parse_category_check_response(user_text, cat)
                self._last_was_category_check = False
                self._pending_category_check_category = None
                if decision == "move_on":
                    self.tree.mark_category_no_need(cat, reason="user confirmed no more requirements in this category")
                    self._categories_done.add(cat)
                    if self._current_category == cat:
                        self._current_category = None
                    self._no_new_in_cat_streak = 0
                else:
                    self._no_new_in_cat_streak = 0
            if self._last_asked_leaf_id:
                self.model_config["__current_leaf_id"] = self._last_asked_leaf_id
            else:
                self.model_config.pop("__current_leaf_id", None)
            pr = parse_user_message_to_leaves(user_text, self.tree.leaf_catalog, self.model_config, self.model_call)
            newly_need_covered = 0
            new_need_in_current_cat = False
            current_cat = self._last_asked_leaf_id.split(".", 1)[0] if self._last_asked_leaf_id else None
            for leaf_id, value, status in pr.answered:
                if self.tree.is_covered(leaf_id):
                    continue
                top_cat = leaf_id.split(".", 1)[0]
                if hasattr(self.tree, "parent_leaf_ids") and leaf_id in self.tree.parent_leaf_ids and status == "no_need":
                    self.tree.mark_subtree_no_need(leaf_id, value or "no_need from parent")
                self.tree.mark_covered(leaf_id, value, status)
                if status == "need":
                    newly_need_covered += 1
                    if current_cat and top_cat == current_cat:
                        new_need_in_current_cat = True
            if newly_need_covered > 0:
                self.state.no_new_info_streak = 0
            else:
                self.state.no_new_info_streak += 1
            # LLM (parse_user_message_to_leaves) judges need/no_need. When user indicates "don't need"
            # (e.g., "I don't care", "just X is enough"), status=no_need → new_need_in_current_cat=False
            # → _no_new_in_cat_streak increments. After streak>=2, category check triggers.
            if current_cat and current_cat not in self._categories_done:
                if new_need_in_current_cat:
                    self._no_new_in_cat_streak = 0
                    if current_cat == "Style":
                        self._style_deferred_count = 0
                else:
                    self._no_new_in_cat_streak += 1
        self.state.last_processed_len = len(conversation_history)

    def _parse_category_check_response(self, user_reply, category):
        interviewer_q = self._format_category_check_question(category)
        usr = get_prompt("ParseUser-Aspect_user").format(interviewer_question=interviewer_q, user_reply=user_reply.strip())
        try:
            data = self.model_call(get_prompt("ParseUser-Aspect_system"), usr, self.model_config, return_json=True)
        except Exception:
            return "continue"
        if not isinstance(data, dict):
            return "continue"
        decision = (data.get("decision") or "").strip().lower()
        return "move_on" if decision == "move_on" else "continue"

    def _format_category_check_question(self, category, is_followup=False):
        if category == "Style":
            return get_prompt("category_check_template_style")
        if category == "Interaction":
            return get_prompt("category_check_template_interaction")
        if category == "Content" and is_followup:
            return get_prompt("category_check_template_content_followup")
        examples = {"Interaction": "search, filter, navigation, buttons", "Content": "data fields, report content, display items", "Style": "colors, layout, theme"}
        category_examples = examples.get(category, "related features")
        return get_prompt("category_check_template").format(category=category, category_examples=category_examples)

    def _layer2_id(self, lid):
        parts = lid.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else lid

    def _score_layer2_units_by_context(self, conversation_history, category, layer2_units):
        if not layer2_units:
            return {}
        history_str = self.build_history(conversation_history, with_note=False)
        if not history_str.strip():
            history_str = "No previous dialogue turns are available yet; only the initial requirements are known."
        units_block_parts = []
        for layer2_id, infos in layer2_units.items():
            desc = [f"  layer2_id={layer2_id}"]
            for info in infos:
                desc.append(f"    - leaf_id={info['leaf_id']}, seed_question={info['seed_question']}")
            units_block_parts.append("\n".join(desc))
        units_block = "\n\n".join(units_block_parts)
        sys_p = get_prompt("Context-awareDynamicRe-ranking_system").format(
            scope_description=f"each requirement \"layer2 unit\" in the \"{category}\" category",
            id_field="layer2_id",
        )
        usr_p = get_prompt("Context-awareDynamicRe-ranking_user").format(
            history_str=history_str,
            scope_intro=f"Below are the requirement \"layer2 units\" in the \"{category}\" category only (each unit = one dimension + its sub-aspects, still undecided). Assign ONE priority score per layer2_id (for the whole unit):",
            candidates_block=units_block,
            id_field="layer2_id",
        )
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

    def _score_layer3_in_layer2(self, conversation_history, layer2_id, layer3_infos):
        if not layer3_infos:
            return {}
        history_str = self.build_history(conversation_history, with_note=False)
        if not history_str.strip():
            history_str = "No previous dialogue turns are available yet; only the initial requirements are known."
        leaves_parts = [f"- leaf_id={info['leaf_id']}, seed_question={info['seed_question']}" for info in layer3_infos]
        leaves_block = "\n".join(leaves_parts)
        sys_p = get_prompt("Context-awareDynamicRe-ranking_system").format(
            scope_description=f"each sub-aspect (leaf) under the dimension \"{layer2_id}\"",
            id_field="leaf_id",
        )
        usr_p = get_prompt("Context-awareDynamicRe-ranking_user").format(
            history_str=history_str,
            scope_intro=f"Below are the sub-aspects (leaves) under dimension \"{layer2_id}\" only (still undecided):",
            candidates_block=leaves_block,
            id_field="leaf_id",
        )
        try:
            resp = self.model_call(sys_p, usr_p, self.model_config, return_json=True)
        except Exception:
            return {}
        scores_out = {}
        if isinstance(resp, dict):
            raw = resp.get("scores") or resp.get("leaf_scores") or []
            if isinstance(raw, list):
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    lid = item.get("leaf_id")
                    if not isinstance(lid, str):
                        continue
                    try:
                        scores_out[lid] = float(item.get("score", 0))
                    except (TypeError, ValueError):
                        continue
        return scores_out

    def _reorder_leaf_order_by_context(self, conversation_history):
        catalog = self.tree.leaf_catalog
        leaf_order = list(self.tree.leaf_order)
        target_cat = None
        for lid in leaf_order:
            if self.tree.should_ask(lid):
                target_cat = lid.split(".", 1)[0]
                break
        if not target_cat:
            return
        candidate_ids = [lid for lid in leaf_order if self.tree.should_ask(lid) and lid.split(".", 1)[0] == target_cat]
        if not candidate_ids:
            return
        candidate_set = set(candidate_ids)
        by_layer2 = {}
        for lid in leaf_order:
            if lid not in candidate_set:
                continue
            l2 = self._layer2_id(lid)
            by_layer2.setdefault(l2, []).append(lid)
        layer2_units = {}
        for layer2_id, lids in by_layer2.items():
            infos = [{"leaf_id": lid, "seed_question": (catalog.get(lid, "") or "").strip()} for lid in lids]
            layer2_units[layer2_id] = infos
        layer2_scores = self._score_layer2_units_by_context(conversation_history, target_cat, layer2_units)
        if not layer2_scores:
            return
        layer2_order = sorted(by_layer2.keys(), key=lambda l2: (-layer2_scores.get(l2, 0.0), min(leaf_order.index(lid) for lid in by_layer2[l2])))
        candidate_ids_sorted = []
        for l2 in layer2_order:
            candidate_ids_sorted.extend(by_layer2[l2])
        new_order = []
        idx = 0
        for lid in leaf_order:
            if lid in candidate_set:
                new_order.append(candidate_ids_sorted[idx])
                idx += 1
            else:
                new_order.append(lid)
        if new_order:
            self.tree.leaf_order = new_order
        first_ask = None
        for lid in self.tree.leaf_order:
            if self.tree.should_ask(lid):
                first_ask = lid
                break
        if not first_ask:
            return
        parts = first_ask.split(".")
        if len(parts) < 3:
            return
        layer2_id = ".".join(parts[:2])
        if layer2_id in self._expanded_layer2:
            return
        prefix = layer2_id + "."
        layer3_in_candidates = [lid for lid in candidate_ids_sorted if lid.startswith(prefix)]
        if not layer3_in_candidates:
            return
        layer3_infos = [{"leaf_id": lid, "seed_question": (catalog.get(lid, "") or "").strip()} for lid in layer3_in_candidates]
        scores_l3 = self._score_layer3_in_layer2(conversation_history, layer2_id, layer3_infos)
        if not scores_l3:
            self._expanded_layer2.add(layer2_id)
            return
        layer3_sorted = sorted(layer3_in_candidates, key=lambda lid: -scores_l3.get(lid, 0.0))
        current_lo = list(self.tree.leaf_order)
        layer2_slice = [lid for lid in current_lo if self._layer2_id(lid) == layer2_id]
        layer2_node = next((lid for lid in layer2_slice if len(lid.split(".")) == 2), None)
        layer3_in_slice = [lid for lid in layer2_slice if len(lid.split(".")) >= 3]
        if not layer3_in_slice:
            self._expanded_layer2.add(layer2_id)
            return
        layer3_rest = [lid for lid in layer3_in_slice if lid not in layer3_sorted]
        new_slice = ([layer2_node] if layer2_node else []) + layer3_sorted + layer3_rest
        new_lo = []
        i = 0
        for lid in current_lo:
            if self._layer2_id(lid) == layer2_id:
                new_lo.append(new_slice[i])
                i += 1
            else:
                new_lo.append(lid)
        self.tree.leaf_order = new_lo
        self._expanded_layer2.add(layer2_id)
