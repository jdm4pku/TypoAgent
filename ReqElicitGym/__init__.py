"""
ReqElicitGym-v7: A Gymnasium environment for requirement elicitation evaluation.

This package provides an evaluation environment where interviewers (LLMs to be evaluated)
interact with simulated users to elicit requirements based on initial requirements
and implicit requirements through natural conversation.
"""

from .env import ReqElicitGym
from .config import (
    ReqElicitGymConfig,
    get_default_config,
)
from .interviewer import Interviewer

__version__ = "0.7.0"

__all__ = [
    "ReqElicitGym",
    "ReqElicitGymConfig",
    "get_default_config",
    "Interviewer",
]
