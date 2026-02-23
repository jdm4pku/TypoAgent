"""Load prompts from TypoAgent/prompt/retriever/*.txt files."""

from pathlib import Path
from typing import Dict


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt" / "retriever"
_CACHE: Dict[str, str] = {}


def _load(name: str) -> str:
    if name not in _CACHE:
        path = _PROMPT_DIR / f"{name}.txt"
        if path.exists():
            _CACHE[name] = path.read_text(encoding="utf-8").strip()
        else:
            _CACHE[name] = ""
    return _CACHE[name]


def get_prompt(name: str) -> str:
    """获取 prompt 内容，name 不带 .txt 后缀。"""
    return _load(name)
