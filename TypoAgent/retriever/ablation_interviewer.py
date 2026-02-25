"""消融实验 Interviewer：基于 AblationTreeAgent，支持初始优先度/上下文优先度/大类门控的开关。"""

from typing import Any, Dict, List

from .ablation_tree_agent import AblationTreeAgent


class AblationInterviewer:
    """消融实验用 Interviewer，使用 AblationTreeAgent。"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        base_url: str = None,
        fixed_tree_path: str = "output/save_tree/LLMTree.auto-p.json",
        tree_percentage: float = 100.0,
        use_initial_priority: bool = True,
        use_context_priority: bool = True,
        use_category_gating: bool = True,
        max_turns_per_category_no_gate: int | None = None,
        cat_check_threshold: int = 2,
        followup_threshold: int = 4,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url
        self.fixed_tree_path = fixed_tree_path
        self.tree_percentage = tree_percentage
        self.use_initial_priority = use_initial_priority
        self.use_context_priority = use_context_priority
        self.use_category_gating = use_category_gating
        self.max_turns_per_category_no_gate = max_turns_per_category_no_gate
        self.cat_check_threshold = cat_check_threshold
        self.followup_threshold = followup_threshold

        self.model_config = {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if base_url:
            self.model_config["base_url"] = base_url

        # 使用 ReqElicitGym 的 model_call 和 build_history
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from ReqElicitGym.env.prompts import model_call
        from ReqElicitGym.env.utils import build_history_into_prompt

        self._tree_agent = AblationTreeAgent(
            fixed_tree_path=self.fixed_tree_path,
            model_config=self.model_config,
            model_call_fn=model_call,
            build_history_fn=build_history_into_prompt,
            use_initial_priority=self.use_initial_priority,
            use_context_priority=self.use_context_priority,
            use_category_gating=self.use_category_gating,
            max_turns_per_category_no_gate=self.max_turns_per_category_no_gate,
            cat_check_threshold=self.cat_check_threshold,
            followup_threshold=self.followup_threshold,
            enable_llm_parsing=True,
            enable_llm_question=True,
            max_ask_per_leaf=2,
            tree_percentage=self.tree_percentage,
        )

    def ask_question(
        self, conversation_history: List[Dict[str, str]], return_usage: bool = False
    ) -> Any:
        q = self._tree_agent.ask_question(conversation_history)
        if return_usage:
            return q, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return q

    def get_config(self) -> Dict[str, Any]:
        return self.model_config.copy()

    def __repr__(self) -> str:
        return (
            f"AblationInterviewer(model={self.model_name}, temperature={self.temperature}, "
            f"initial_priority={self.use_initial_priority} "
            f"context_priority={self.use_context_priority} category_gating={self.use_category_gating})"
        )
