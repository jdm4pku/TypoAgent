"""Base RequirementTreeAgent: static order only."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from .implicit_topic_parser import parse_user_message_to_leaves
from .prompt_loader import get_prompt
from .static_requirement_tree import StaticRequirementTree

@dataclass
class AgentState:
    last_processed_len: int = 0
    turns_asked: int = 0
    no_new_info_streak: int = 0

class RequirementTreeAgent:
    def __init__(self, fixed_tree_path, model_config, model_call_fn, build_history_fn,
                 enable_llm_parsing=True, enable_llm_question=True, max_ask_per_leaf=2,
                 max_turns=20, no_new_info_patience=3, min_turns_before_early_stop=3, tree_percentage=100.0):
        self.fixed_tree_path = fixed_tree_path
        self.model_config = model_config
        self.model_call = model_call_fn
        self.build_history = build_history_fn
        self.enable_llm_parsing = enable_llm_parsing
        self.enable_llm_question = enable_llm_question
        self.tree = StaticRequirementTree(fixed_tree_path=fixed_tree_path, max_ask_per_leaf=max_ask_per_leaf, percentage=tree_percentage)
        self.state = AgentState()
        self._last_initial_req = None
        self._last_asked_leaf_id = None
        self.max_turns = max_turns
        self.no_new_info_patience = no_new_info_patience
        self.min_turns_before_early_stop = min_turns_before_early_stop
        self._current_category = None
        self._no_new_in_cat_streak = 0
        self._categories_done = set()

    def reset(self):
        self.tree.reset_state()
        self.state = AgentState(last_processed_len=0, turns_asked=0, no_new_info_streak=0)
        self._last_initial_req = None
        self._last_asked_leaf_id = None
        self._current_category = None
        self._no_new_in_cat_streak = 0
        self._categories_done = set()

    def ask_question(self, conversation_history):
        self._maybe_reset_episode(conversation_history)
        self._update_state_from_history(conversation_history)
        if self.tree.all_determined():
            return self._finish(conversation_history)
        if self.state.turns_asked >= self.max_turns:
            return self._finish(conversation_history)
        leaf_id = self.tree.next_leaf()
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
            leaf_id2 = self.tree.next_leaf()
            if leaf_id2 is None:
                return self._finish(conversation_history)
            seed2 = self.tree.get_seed_question(leaf_id2)
            question = self._ask_leaf(conversation_history, leaf_id2, seed2)
            self.tree.mark_asked(leaf_id2)
            self._last_asked_leaf_id = leaf_id2
            self.state.turns_asked += 1
            return question or seed2
        self.tree.mark_asked(leaf_id)
        self._last_asked_leaf_id = leaf_id
        self.state.turns_asked += 1
        return question

    def _maybe_reset_episode(self, conversation_history):
        if not conversation_history:
            self.reset()
            return
        if len(conversation_history) == 1 and conversation_history[0].get("role") == "user":
            initial = (conversation_history[0].get("content") or "").strip()
            if self._last_initial_req != initial:
                self.reset()
                self._last_initial_req = initial
            return
        if len(conversation_history) < self.state.last_processed_len:
            self.reset()

    def _update_state_from_history(self, conversation_history):
        start = max(self.state.last_processed_len, 0)
        new_msgs = conversation_history[start:]
        for msg in new_msgs:
            if msg.get("role") != "user":
                continue
            user_text = (msg.get("content") or "").strip()
            if not user_text:
                continue
            if self._last_asked_leaf_id:
                self.model_config["__current_leaf_id"] = self._last_asked_leaf_id
            else:
                self.model_config.pop("__current_leaf_id", None)
            pr = parse_user_message_to_leaves(user_text, self.tree.leaf_catalog, self.model_config, self.model_call)
            newly_need_covered = 0
            new_need_in_current_cat = False
            current_cat = self._last_asked_leaf_id.split(".", 1)[0] if self._last_asked_leaf_id else None
            asked = self._last_asked_leaf_id or ""

            def _is_asked_branch(lid):
                return lid == asked or lid.startswith(asked + ".") or (asked.startswith(lid + ".") if lid else False)

            has_any_answered_for_asked_branch = False
            got_no_need_for_asked = False
            for leaf_id, value, status in pr.answered:
                if _is_asked_branch(leaf_id):
                    has_any_answered_for_asked_branch = True
                    if status == "no_need":
                        got_no_need_for_asked = True
                if self.tree.is_covered(leaf_id):
                    continue
                top_cat = leaf_id.split(".", 1)[0]
                if leaf_id in self.tree.parent_leaf_ids and status == "no_need":
                    self.tree.mark_subtree_no_need(leaf_id, value or "no_need from parent")
                self.tree.mark_covered(leaf_id, value, status)
                if status == "need":
                    newly_need_covered += 1
                    if current_cat and top_cat == current_cat and _is_asked_branch(leaf_id):
                        new_need_in_current_cat = True
            if not has_any_answered_for_asked_branch and not new_need_in_current_cat and asked:
                got_no_need_for_asked = True
            if newly_need_covered > 0:
                self.state.no_new_info_streak = 0
            else:
                self.state.no_new_info_streak += 1
            if current_cat and current_cat not in self._categories_done:
                if new_need_in_current_cat:
                    self._no_new_in_cat_streak = 0
                elif got_no_need_for_asked:
                    self._no_new_in_cat_streak += 1
                if self._no_new_in_cat_streak >= 3:
                    self.tree.mark_category_no_need(current_cat, reason="switch to next category after 3 no-new-need turns")
                    self._categories_done.add(current_cat)
                    self._current_category = None
                    self._no_new_in_cat_streak = 0
        self.state.last_processed_len = len(conversation_history)

    def _ask_leaf(self, conversation_history, leaf_id, seed_question):
        history = self.build_history(conversation_history, with_note=False)
        if not self.enable_llm_question:
            return seed_question
        user_prompt = get_prompt("GenerateQuestion_user").format(history=history, leaf_id=leaf_id, seed_question=seed_question)
        text = self.model_call(get_prompt("GenerateQuestion_system"), user_prompt, self.model_config, return_json=False)
        return (text or "").strip()

    def _finish(self, conversation_history):
        history = self.build_history(conversation_history, with_note=False)
        user_prompt = get_prompt("finish_user").format(history=history)
        text = self.model_call(get_prompt("finish_system"), user_prompt, self.model_config, return_json=False)
        return (text or "").strip() or "I have gathered enough information. The user requirements list is as follows:\n\nFeature1: As a user, I want to ... (TBD)"
