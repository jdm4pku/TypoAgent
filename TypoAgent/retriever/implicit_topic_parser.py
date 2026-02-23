"""Parse user messages against the fixed tree.
Need/no_need is determined by LLM based ONLY on user's answer:
- need: user expresses desire/preference (e.g. "I'd like", "I want")
- no_need: user expresses no care/no need (e.g. "I don't care about", "Nothing else comes to mind")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .prompt_loader import get_prompt


@dataclass
class ParseResult:
    answered: List[Tuple[str, str, str]]  # (leaf_id, value, status)
    related: List[str]
    notes: str = ""


def parse_user_message_to_leaves(
    user_message: str,
    leaf_catalog: Dict[str, str],
    model_config: Dict,
    model_call_fn,
) -> ParseResult:
    """Parse user answer to need/no_need. Uses LLM to judge based ONLY on user's answer
    whether this round's question hit the user's intended need (need) or not (no_need)."""
    if not user_message.strip():
        return ParseResult(answered=[], related=[], notes="")

    current_leaf_id = (model_config or {}).get("__current_leaf_id")
    if current_leaf_id is not None:
        current_leaf_id = str(current_leaf_id).strip() or None

    if current_leaf_id and current_leaf_id in leaf_catalog:
        # New logic: use LLM to judge need/no_need based ONLY on user's answer
        interviewer_question = (leaf_catalog.get(current_leaf_id) or "").strip()
        sys_prompt = get_prompt("ParseUser-DimensionSlot_system")
        usr_prompt = get_prompt("ParseUser-DimensionSlot_user").format(
            interviewer_question=interviewer_question or "(no question provided)",
            user_answer=user_message.strip(),
            leaf_id=current_leaf_id,
        )
        try:
            data = model_call_fn(sys_prompt, usr_prompt, model_config, return_json=True)
        except Exception:
            data = {}

        answered: List[Tuple[str, str, str]] = []
        if isinstance(data, dict):
            lid = str(data.get("leaf_id", "")).strip()
            val = str(data.get("value", "")).strip()
            st = str(data.get("status", "")).strip().lower()
            if lid == current_leaf_id and st in ("need", "no_need"):
                answered = [(lid, val or ("no answer provided" if st == "no_need" else ""), st)]

        if not answered:
            answered = [(current_leaf_id, "no answer provided", "no_need")]

        return ParseResult(answered=answered, related=[], notes="")
    else:
        return ParseResult(answered=[], related=[], notes="")
