"""
LLMREI-Long baseline interviewer for requirement elicitation.
加载 prompt 从 baseline/prompt/long/ 下的 txt 文件，复用 ReqElicitGym 的 model_call 与 build_history_into_prompt。
"""

from pathlib import Path
from typing import Any, Dict, List

_PROMPT_DIR = Path(__file__).resolve().parent / "prompt" / "long"


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


class Interviewer:
    """LLMREI-Long 风格 Interviewer，使用长系统 prompt。"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        base_url: str = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url

        self.model_config = {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if base_url:
            self.model_config["base_url"] = base_url

        self._system_prompt = _load_prompt("system")
        self._user_tmpl = _load_prompt("user")

    def ask_question(
        self, conversation_history: List[Dict[str, str]], return_usage: bool = False
    ) -> Any:
        from ReqElicitGym.env.prompts import model_call
        from ReqElicitGym.env.utils import build_history_into_prompt

        history_str = build_history_into_prompt(conversation_history, with_note=False)
        if not history_str:
            history_str = "User: [Initial requirements]"

        user_prompt = self._user_tmpl.format(history_str=history_str)

        if return_usage:
            response_text, usage_info = model_call(
                self._system_prompt,
                user_prompt,
                self.model_config,
                return_json=False,
                return_usage=True,
            )
            return (response_text if response_text else "", usage_info)
        else:
            response_text = model_call(
                self._system_prompt,
                user_prompt,
                self.model_config,
                return_json=False,
                return_usage=False,
            )
            return response_text if response_text else ""

    def get_config(self) -> Dict[str, Any]:
        return self.model_config.copy()

    def __repr__(self) -> str:
        return f"Interviewer(model={self.model_name}, style=LLMREI-Long)"
