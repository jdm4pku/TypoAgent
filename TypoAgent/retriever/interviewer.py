"""TypoAgentInterviewer: 基于树的 Interviewer，与 interviewer_DynamicTree_shuffle_priority 等效。"""

from typing import Any, Dict, List

from .dynamic_shuffle_agent import DynamicShuffleAgent


class TypoAgentInterviewer:
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
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url
        self.fixed_tree_path = fixed_tree_path
        self.tree_percentage = tree_percentage

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

        self._tree_agent = DynamicShuffleAgent(
            fixed_tree_path=self.fixed_tree_path,
            model_config=self.model_config,
            model_call_fn=model_call,
            build_history_fn=build_history_into_prompt,
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
        return f"TypoAgentInterviewer(model={self.model_name}, tree=DynamicTree_shuffle_priority)"
