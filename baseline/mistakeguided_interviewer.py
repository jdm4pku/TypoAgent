"""
MistakeGuided / Comp3 Interviewer: 基于 Requirements-Elicitation-Follow-Up-Question-Generation
方法的 Interviewer 实现。

参考: https://github.com/anmolsinghal98/Requirements-Elicitation-Follow-Up-Question-Generation
论文: Y. Shen, A. Singhal, T.D. Breaux (2025). "Requirements Elicitation Follow-up Question Generation."
      33rd IEEE International Requirements Engineering Conference.

该方法使用 GPT 根据对话上下文生成跟进问题（Follow-up Question），适用于需求启发访谈。
本实现仿照 long_interviewer / short_interviewer，将 prompts 拆分为
baseline/prompt/mistakeguided/ 目录下的 txt 文件，并复用 ReqElicitGym 的 model_call 与
build_history_into_prompt。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ReqElicitGym.env.prompts import model_call
from ReqElicitGym.env.utils import build_history_into_prompt


_PROMPT_DIR = Path(__file__).resolve().parent / "prompt" / "mistakeguided"


def _load_prompt(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


class Interviewer:
    """
    MistakeGuided（Comp3）Interviewer：使用 Requirements-Elicitation-Follow-Up-Question-Generation
    方法生成跟进问题。

    实现与 ReqElicitGym 兼容的 Interviewer 接口：
    - ask_question(conversation_history, return_usage=False) -> question or (question, usage_info)
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        enable_finish: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_finish = enable_finish

        self.model_config = {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if base_url:
            self.model_config["base_url"] = base_url

        # 加载 prompts（与 long_interviewer / short_interviewer 风格保持一致）
        self._system_prompt = _load_prompt("system")
        self._user_tmpl = _load_prompt("user")
        self._user_with_finish_tmpl = _load_prompt("user_with_finish")

    def ask_question(
        self, conversation_history: List[Dict[str, str]], return_usage: bool = False
    ) -> Any:
        history_str = build_history_into_prompt(conversation_history, with_note=False)
        if not history_str:
            history_str = "User: [Initial requirements - no prior conversation]"

        # 从首条用户消息推断访谈领域（软件需求）
        interview_domain = "a software system"
        if conversation_history:
            first_user = next(
                (e.get("content", "") for e in conversation_history if e.get("role") == "user"),
                ""
            )
            if first_user:
                # 用首句摘要作为领域描述，截断避免过长
                interview_domain = first_user[:200] + ("..." if len(first_user) > 200 else "")

        # 根据是否允许 finish 动作选择不同的 user prompt 模版
        if self.enable_finish:
            user_prompt = self._user_with_finish_tmpl.format(
                interview_domain=interview_domain,
                interview_turns=history_str,
            )
        else:
            user_prompt = self._user_tmpl.format(
                interview_domain=interview_domain,
                interview_turns=history_str,
            )

        if return_usage:
            response_text, usage_info = model_call(
                self._system_prompt,
                user_prompt,
                self.model_config,
                return_json=False,
                return_usage=True,
            )
            return (response_text.strip() if response_text else "", usage_info)
        else:
            response_text = model_call(
                self._system_prompt,
                user_prompt,
                self.model_config,
                return_json=False,
                return_usage=False,
            )
            return response_text.strip() if response_text else ""

    def get_config(self) -> Dict[str, Any]:
        """返回模型配置字典。"""
        return self.model_config.copy()

    def __repr__(self) -> str:
        return f"Interviewer(model={self.model_name}, method=Requirements-Elicitation-Follow-Up)"
