"""消融实验 Agent：在 DynamicShuffleAgent 基础上通过开关控制三项能力。

- use_initial_priority: 是否使用「初始需求优先度打分排序」
- use_context_priority: 是否使用「过程中上下文打分排序」
- use_category_gating: 是否使用「大类门控剪枝」（无新增时问「此大类还要吗」并整类跳过）
"""

from __future__ import annotations

from typing import Dict, List

from .dynamic_shuffle_agent import DynamicShuffleAgent


class AblationTreeAgent(DynamicShuffleAgent):
    """继承 DynamicShuffleAgent，通过三个布尔开关做消融。"""

    def __init__(
        self,
        fixed_tree_path: str,
        model_config: Dict,
        model_call_fn=None,
        build_history_fn=None,
        use_initial_priority: bool = True,
        use_context_priority: bool = True,
        use_category_gating: bool = True,
        max_turns_per_category_no_gate: int | None = None,
        enable_llm_parsing: bool = True,
        enable_llm_question: bool = True,
        max_ask_per_leaf: int = 2,
        max_turns: int = 20,
        no_new_info_patience: int = 3,
        min_turns_before_early_stop: int = 3,
        tree_percentage: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(
            fixed_tree_path=fixed_tree_path,
            model_config=model_config,
            model_call_fn=model_call_fn,
            build_history_fn=build_history_fn,
            enable_llm_parsing=enable_llm_parsing,
            enable_llm_question=enable_llm_question,
            max_ask_per_leaf=max_ask_per_leaf,
            max_turns=max_turns,
            no_new_info_patience=no_new_info_patience,
            min_turns_before_early_stop=min_turns_before_early_stop,
            tree_percentage=tree_percentage,
            **kwargs,
        )
        self._use_initial_priority = use_initial_priority
        self._use_context_priority = use_context_priority
        self._use_category_gating = use_category_gating
        # 在关闭大类门控时，每个大类最多允许提问多少次；为 None 表示不额外限制
        self._max_turns_per_category_no_gate = max_turns_per_category_no_gate
        self._cat_asked_turns: Dict[str, int] = {}

    def reset(self) -> None:
        """重置时顺便清空每类计数。"""
        super().reset()
        self._cat_asked_turns = {}

    def _maybe_reset_episode(self, conversation_history: List[Dict[str, str]]) -> None:
        """与 StaticShuffleAgent 一致，但仅在 use_initial_priority 时做初始优先度重排。"""
        if not conversation_history:
            self.reset()
            return

        if len(conversation_history) == 1 and conversation_history[0].get("role") == "user":
            initial = (conversation_history[0].get("content") or "").strip()
            if self._init_state_last_initial != initial:
                self.reset()
                self._init_state_last_initial = initial
                if self._use_initial_priority:
                    self._reorder_leaf_order_by_initial(initial)
            return

        if len(conversation_history) < self.state.last_processed_len:
            self.reset()

    def ask_question(self, conversation_history: List[Dict[str, str]]) -> str:
        """仅当 use_context_priority 时做上下文重排；仅当 use_category_gating 时做大类门控。"""
        self._maybe_reset_episode(conversation_history)
        self._update_state_from_history(conversation_history)

        # 先判断结束和大类确认，避免不必要的上下文重排 LLM 调用
        if self.tree.all_determined() and len(self._categories_asked) >= 3:
            return self._finish(conversation_history)
        if self.state.turns_asked >= self.max_turns:
            return self._finish(conversation_history)

        current_cat = getattr(self, "_current_category", None)
        cat_check_threshold = 3 if current_cat == "Content" else 2
        followup_threshold = 5 if current_cat == "Content" else 4
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
            and current_cat in getattr(self, "_category_check_asked", set())
            and current_cat not in getattr(self, "_category_followup_asked", set())
        )
        if self._use_category_gating and (first_check or followup_check):
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

        # 需要选下一个 leaf 时，再按上下文动态重排（仅 use_context_priority 时）
        if self._use_context_priority:
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

        # 在「没有大类门控」时，为防止单个大类遍历过多叶子，增加一个固定上限
        if not self._use_category_gating and self._max_turns_per_category_no_gate is not None:
            cnt = self._cat_asked_turns.get(top_cat, 0) + 1
            self._cat_asked_turns[top_cat] = cnt
            if (
                cnt >= self._max_turns_per_category_no_gate
                and top_cat not in getattr(self, "_categories_done", set())
            ):
                try:
                    self.tree.mark_category_no_need(
                        top_cat,
                        reason=(
                            "reached fixed max question count per category "
                            "when category gating is disabled"
                        ),
                    )
                    self._categories_done.add(top_cat)
                except Exception:
                    pass

        return question
