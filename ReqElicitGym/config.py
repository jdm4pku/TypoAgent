"""
Configuration management for ReqElicitGym environments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os

@dataclass
class ReqElicitGymConfig:
    """Configuration class for ReqElicitGym environments."""

    # Data configuration - 简化：直接指定数据文件路径
    data_path: str = "data/test.json"  # 数据文件路径

    # Model configuration for judge
    judge_api_key: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_model_name: str = "gpt-5.1"
    judge_temperature: float = 0
    judge_max_tokens: int = 1024
    judge_timeout: float = 30.0
    
    # Model configuration for simulated user (GPT-5.1)
    user_api_key: Optional[str] = None
    user_base_url: Optional[str] = None
    user_model_name: str = "gpt-5.1"
    user_temperature: float = 0.7
    user_max_tokens: int = 1024
    user_timeout: float = 30.0

    # User answer quality levels: "high", "medium", "low"
    user_answer_quality: str = "high"

    # Environment configuration
    max_steps: int = 20
    max_turns: int = 40  # Maximum conversation turns

    # Tracking configuration  
    track_conversation_history: bool = True
    track_elicit_state: bool = True

    # Environment behavior
    verbose: bool = False
    seed: Optional[int] = None
    
    # Output file paths (optional, if None, must be provided when calling save methods)
    evaluation_result_path: Optional[str] = None  # Path for evaluation results JSON file
    conversation_result_path: Optional[str] = None  # Path for conversation results JSON file

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.judge_api_key is None:
            raise ValueError("judge_api_key must be provided")
        if self.user_api_key is None:
            raise ValueError("user_api_key must be provided")
        if self.user_answer_quality not in ["high", "medium", "low"]:
            raise ValueError(f"user_answer_quality must be one of ['high', 'medium', 'low'], got {self.user_answer_quality}")
        if not self.data_path:
            raise ValueError("data_path must be provided")

    def validate(self):
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.judge_temperature < 0:
            raise ValueError("judge_temperature must be non-negative")
        if self.user_temperature < 0:
            raise ValueError("user_temperature must be non-negative")
        if self.judge_max_tokens <= 0:
            raise ValueError("judge_max_tokens must be positive")
        if self.user_max_tokens <= 0:
            raise ValueError("user_max_tokens must be positive")
        if self.judge_timeout <= 0:
            raise ValueError("judge_timeout must be positive")
        if self.user_timeout <= 0:
            raise ValueError("user_timeout must be positive")
        
        # 检查数据文件是否存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReqElicitGymConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

def get_default_config() -> ReqElicitGymConfig:
    """Get the default ReqElicitGym configuration."""
    return ReqElicitGymConfig()
