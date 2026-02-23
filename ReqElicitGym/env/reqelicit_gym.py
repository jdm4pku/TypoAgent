"""
ReqElicitEnv: A Gymnasium environment for requirement elicitation evaluation.

This module provides a Gymnasium environment where a interviewer agent interact with a simulated
user to elicit requirements and provide user requirements list (URL).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import random
import json

from .prompts import evaluate_action
from .task_data import load_tasks
from ..config import ReqElicitGymConfig, get_default_config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..interviewer import Interviewer

class ReqElicitGym(gym.Env):
    """
    ReqElicitGym Environment for requirement elicitation evaluation.
    
    This environment simulates a conversation between an interviewer agent (to be evaluated)
    and a simulated user (GPT-5.1) where the interviewer agent needs to elicit requirements
    and write down the user requirements list (URL).
    """
    
    def __init__(self, config: ReqElicitGymConfig = None):
        """Initialize the ReqElicitGym environment."""
        super().__init__()

        # Set configuration
        self.config = config if config is not None else get_default_config()
        self.config.validate()

        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Load all tasks from data file (简化：直接加载所有任务，按顺序运行)
        self.current_task_index = 0
        self._load_tasks()
        
        # Initialize global statistics for evaluation across all tasks
        self.global_stats = {
            "task_results": [],  # List of task-level statistics
            "total_tasks": len(self.tasks) if hasattr(self, 'tasks') else 0,
            "task_step_records": [],  # List of step-by-step records for each task
            "conversation_turns": [],  # List of conversation turns for all tasks
        }
        
        # Per-task tracking variables
        self.current_task_total_requirements = 0
        self.current_task_elicited = 0  # Requirements elicited by interviewer
        
        # Track requirements by aspect type (Interaction, Content, Style)
        self.current_task_requirements_by_aspect = {
            "Interaction": {"total": 0, "elicited": 0},
            "Content": {"total": 0, "elicited": 0},
            "Style": {"total": 0, "elicited": 0}
        }
        
        # Step-by-step recording for current task
        self.current_task_step_records = []  # Records for each step in current task
        self.current_task_hit_sequence = []  # Hit sequence H=(h_1,...,h_n) for TKQR calculation
        self.current_task_conversation_turns = []  # Conversation turns for current task
        
        # Action type effectiveness tracking
        # Dictionary: {action_type: {"total": count, "effective": count}}
        self.current_task_action_stats = {}
        
        # Token usage tracking for current task (interviewer question generation)
        self.current_task_token_cost = 0  # Total tokens used for generating questions in current task
        
        # Store interviewer model name for saving results
        self.interviewer_model_name = None

        # Define action and observation spaces
        self.action_space = spaces.Text(max_length=1000)
        
        # Observation space is a dictionary
        self.observation_space = spaces.Dict({
            "task_description": spaces.Text(max_length=5000),
            "goal": spaces.Text(max_length=500),
            "feedback": spaces.Text(max_length=5000),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=int),
            "episode_complete": spaces.Discrete(2),
            "total_requirements": spaces.Box(low=0, high=100, shape=(), dtype=int),
            "remaining_requirements": spaces.Box(low=0, high=100, shape=(), dtype=int),
            "elicitation_ratio": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
            "conversation_history": spaces.Text(max_length=10000),
        })

        # Initialize state variables
        self.reset()

    def _load_tasks(self):
        """Load all tasks from data file (简化：直接加载所有任务)."""
        self.tasks = load_tasks(self.config.data_path)
        if self.config.verbose:
            print(f"Loaded {len(self.tasks)} tasks from {self.config.data_path}")
        
    def _get_next_task(self):
        """Get the next task in sequence (按顺序返回任务)."""
        if self.current_task_index >= len(self.tasks):
            # 如果所有任务都跑完了，抛出异常
            raise StopIteration(f"All tasks completed. Total tasks: {len(self.tasks)}")
        
        task = self.tasks[self.current_task_index]
        self.current_task_index += 1
        return task
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Observation and info dictionary: Tuple of (observation, info)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_complete = False
        self.step_count = 0

        # Reset history tracking
        self.action_history = []
        self.conversation_history = []
        self.elicited_requirements = []

        # Get a new task (按顺序获取下一个任务)
        try:
            self.current_task = self._get_next_task()
        except StopIteration as e:
            # 所有任务已完成
            if self.config.verbose:
                print(f"所有任务已完成: {e}")
            # 返回一个表示任务已完成的观察和信息
            observation = {
                "task_description": "All tasks completed",
                "goal": "All tasks completed",
                "feedback": "All tasks have been completed. No more tasks available.",
                "step_count": 0,
                "episode_complete": True,
                "total_requirements": 0,
                "remaining_requirements": 0,
                "elicitation_ratio": 0.0,
                "conversation_history": "",
            }
            info = {
                "task_id": "",
                "requirements_summary": [],
                "action_history": [],
                "conversation_history": [],
                "elicited_requirements": [],
                "all_tasks_completed": True,
            }
            return observation, info
        
        # Add initial user requirement as first user message
        initial_requirements = self.current_task.get("initial_requirements", "")
        if initial_requirements:
            self.conversation_history.append({
                "role": "user",
                "content": initial_requirements
            })
        
        # Initialize requirements tracking
        self._initialize_requirements()  

        # Initialize step counter
        self.step_count = 0
        self.episode_complete = False
        
        # Initialize per-task statistics
        self.current_task_total_requirements = len(self.remaining_requirements)
        self.current_task_elicited = 0
        
        # Initialize requirements by aspect type
        self.current_task_requirements_by_aspect = {
            "Interaction": {"total": 0, "elicited": 0},
            "Content": {"total": 0, "elicited": 0},
            "Style": {"total": 0, "elicited": 0}
        }
        
        # Count total requirements by aspect type
        for req in self.remaining_requirements:
            aspect = req.get("aspect", "")
            if aspect in self.current_task_requirements_by_aspect:
                self.current_task_requirements_by_aspect[aspect]["total"] += 1
        
        # Initialize step-by-step recording for current task
        self.current_task_step_records = []
        self.current_task_hit_sequence = []  # Initialize hit sequence for TKQR
        self.current_task_action_stats = {}  # Initialize action type statistics
        self.current_task_conversation_turns = []  # Initialize conversation turns for current task
        self.current_task_token_cost = 0  # Initialize token cost for current task
        
        # Record initial state (step 0)
        self._record_step_statistics()
        
        # Build observation
        # v7 格式：使用 name, application_type, initial_requirements 等字段
        task_description = f"System Name: {self.current_task.get('name', 'N/A')}\n"
        task_description += f"Application Type: {self.current_task.get('application_type', 'N/A')}\n"
        task_description += f"Initial Requirements: {self.current_task.get('initial_requirements', 'N/A')}"
        
        observation = {
            "task_description": task_description,
            "goal": "Elicit requirements and write user requirements list if elicit enough requirements",
            "feedback": self.current_task.get("initial_requirements", "Let's start the conversation!"),
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_requirements": len(self.remaining_requirements) + len(self.elicited_requirements),
            "remaining_requirements": len(self.remaining_requirements),
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "conversation_history": self._build_conversation_history_str(),
        }

        # Build info dictionary
        info = {
            "task_id": self.current_task.get("id", ""),
            "requirements_summary": self._get_remaining_requirements_summary(),
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "elicited_requirements": self.elicited_requirements.copy(),
        }
        
        if self.config.verbose:
            print(f"🎯 New Episode Started")
            print(f"Task ID: {info['task_id']}")
            print(f"Task Index: {self.current_task_index - 1}/{len(self.tasks)}")
            print(f"Total Requirements: {observation['total_requirements']}")
            print(f"Task Description: {observation['task_description'][:100]}...")

        return observation, info

    def _initialize_requirements(self):
        """Initialize requirements tracking from the current task (适配 v7 格式)."""
        self.remaining_requirements = []

        # Extract implicit requirements from task data (v7 格式: "Implicit Requirements")
        implicit_requirements = self.current_task.get("Implicit Requirements", [])

        implicit_req_id_counter = 1
        for req_data in implicit_requirements:
            # v7 格式: {"Aspect": "...", "RequirementText": "..."}，没有 "Corresponding User Story" 字段
            aspect = req_data.get("Aspect", "")
            requirement_text = req_data.get("RequirementText", "")
            
            # 根据 Aspect 确定维度：Interaction/Content -> FR, Style -> NFR
            # 也可以根据实际需求调整
            dimension = "NFR" if aspect == "Style" else "FR"
            
            implicit_req = {
                "id": f"IR{implicit_req_id_counter}",
                "aspect": aspect,
                "requirement": requirement_text,  # 使用 requirement 字段名以保持兼容
                "dimension": dimension,
                "elicited": False
            }
            self.remaining_requirements.append(implicit_req)
            implicit_req_id_counter += 1
        
        if not self.remaining_requirements:
            self.remaining_requirements = []

    def _calculate_elicitation_ratio(self) -> float:
        """Calculate the ratio of elicited requirements."""
        total_requirements = len(self.remaining_requirements) + len(self.elicited_requirements)
        if total_requirements == 0:
            return 0.0
        return len(self.elicited_requirements) / total_requirements

    def _get_remaining_requirements_summary(self) -> List[str]:
        """Get a summary of remaining requirements."""
        return [
            f"{req['id']}: {req.get('aspect', '')}-{req.get('requirement', '')[:50]}..."
            for req in self.remaining_requirements
        ]
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The interviewer's question/action (string)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        

        # add interviewer's action to conversation history
        self.conversation_history.append({
            "role": "interviewer",
            "content": action
        })

        # judge model config
        judge_model_config = {
            "api_key": self.config.judge_api_key,
            "base_url": self.config.judge_base_url,
            "model_name": self.config.judge_model_name,
            "temperature": self.config.judge_temperature,
            "max_tokens": self.config.judge_max_tokens,
            "timeout": self.config.judge_timeout,
        }
        
        # create model config dict for evaluation
        user_simulator_config = {
            "api_key": self.config.user_api_key,
            "base_url": self.config.user_base_url,
            "model_name": self.config.user_model_name,
            "temperature": self.config.user_temperature,
            "max_tokens": self.config.user_max_tokens,
            "timeout": self.config.user_timeout,
        }

        user_quality_level = self.config.user_answer_quality

        # Evaluate action (reward is not used, but kept for API compatibility)
        user_response, elicited_requirements, _, judgement = evaluate_action(
            action=action,
            task=self.current_task,
            judge_model_config = judge_model_config,
            user_simulator_config = user_simulator_config,
            conversation_history = self.conversation_history[:-1],
            remaining_requirements = self.remaining_requirements,
            user_quality_level = user_quality_level,
        )

        # Update elicited requirements
        if elicited_requirements:
            for req_id in elicited_requirements:
                for req in self.remaining_requirements:
                    if req.get("id") == req_id and not req.get("elicited", False):
                        req["elicited"] = True
                        self.elicited_requirements.append(req.copy())
                        
                        # Update aspect-specific statistics
                        aspect = req.get("aspect", "")
                        if aspect in self.current_task_requirements_by_aspect:
                            self.current_task_requirements_by_aspect[aspect]["elicited"] += 1
                        break
            
            # Remove elicited requirements from remaining list
            self.remaining_requirements = [
                req for req in self.remaining_requirements
                if not req.get("elicited", False)
            ]
        
        # Update per-task statistics
        elicited_in_this_step = len(elicited_requirements) if elicited_requirements else 0
        self.current_task_elicited += elicited_in_this_step
        
        # Record hit for TKQR calculation (h_i = 1 if hit implicit requirements, 0 otherwise)
        is_hit = judgement.get("is_relevant_to_implied_requirements", False)
        self.current_task_hit_sequence.append(1 if is_hit else 0)
        
        # Track action type effectiveness
        action_type = judgement.get("action_type", "unknown")
        if action_type not in self.current_task_action_stats:
            self.current_task_action_stats[action_type] = {"total": 0, "effective": 0}
        self.current_task_action_stats[action_type]["total"] += 1
        if is_hit:
            self.current_task_action_stats[action_type]["effective"] += 1
        
        # Record step statistics (after updating counts)
        self._record_step_statistics()
        
        # Use judgement as action_info
        action_info = judgement.copy()
        action_info["elicited_requirements"] = elicited_requirements
        action_info["user_response"] = user_response

        # Record conversation turn (turn number is step_count + 1 because step_count will be incremented after)
        # Calculate current elicitation_ratio after updating elicited requirements
        current_elicitation_ratio = self._calculate_elicitation_ratio()
        conversation_turn = {
            "turn": self.step_count + 1,
            "interviewer": action,
            "user": user_response,
            "action_type": action_info.get("action_type", "unknown"),
            "is_relevant_to_url": action_info.get("is_relevant_to_implied_requirements", False),
            "elicitation_ratio": current_elicitation_ratio,  # Add elicitation ratio after this turn
        }
        self.current_task_conversation_turns.append(conversation_turn)

        # Add user response to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_response
        })

        # Update step count
        self.step_count +=1
        
        # Track action history
        self.action_history.append(action)

        if action_info.get("action_type") == "finish":
            self.episode_complete = True
            terminated = True
            truncated = False
            # Record task statistics when episode completes
            self._record_task_statistics()
        elif self.step_count >= self.config.max_steps:
            self.episode_complete = True
            terminated = False
            truncated = True
            # Record task statistics when episode is truncated
            self._record_task_statistics()
        else:
            terminated = False
            truncated = False

        # v7 格式：使用 name, application_type, initial_requirements 等字段
        task_description = f"System Name: {self.current_task.get('name', 'N/A')}\n"
        task_description += f"Application Type: {self.current_task.get('application_type', 'N/A')}\n"
        task_description += f"Initial Requirements: {self.current_task.get('initial_requirements', 'N/A')}"
        
        observation = {
            "task_description": task_description,
            "goal": "Elicit requirements and write user requirements list if elicit enough requirements",
            "feedback": user_response,
            "step_count": self.step_count,
            "episode_complete": int(self.episode_complete),
            "total_requirements": len(self.remaining_requirements) + len(self.elicited_requirements),
            "remaining_requirements": len(self.remaining_requirements),
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "conversation_history": self._build_conversation_history_str(),
        }

        info = {
            "task_id": self.current_task.get("id", ""),
            "requirements_summary": self._get_remaining_requirements_summary(),
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "elicited_requirements": self.elicited_requirements.copy(),
            "action_info": action_info,
        }

        # Return 0.0 for reward (required by Gymnasium interface, but not used)
        return observation, 0.0, terminated, truncated, info

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history."""
        return self.conversation_history.copy()
    
    def _build_conversation_history_str(self) -> str:
        """Build conversation history as string."""
        history_str = ""
        for entry in self.conversation_history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "interviewer":
                history_str += f"Interviewer: {content}\n\n"
            elif role == "user":
                history_str += f"User: {content}\n\n"
        return history_str.strip()
    
    def _calculate_tkqr(self) -> float:
        """
        Calculate Turn-discounted Key Question Rate (TKQR).
        
        Returns:
            TKQR value in [0, 1]
        """
        import math
        
        n = len(self.current_task_hit_sequence)  # Total number of dialogue turns
        K = self.current_task_total_requirements  # Total number of implicit requirements
        
        if n == 0 or K == 0:
            return 0.0
        
        # Calculate DCG_n = sum(i=1 to n) h_i / log_2(i+1)
        dcg = 0.0
        for i, h_i in enumerate(self.current_task_hit_sequence, start=1):
            if h_i == 1:
                dcg += 1.0 / math.log2(i + 1)
        
        # Calculate IDCG_n = sum(i=1 to min(n,K)) 1 / log_2(i+1)
        idcg = 0.0
        for i in range(1, min(n, K) + 1):
            idcg += 1.0 / math.log2(i + 1)
        
        # Calculate TKQR = DCG_n / IDCG_n
        if idcg == 0:
            return 0.0
        
        tkqr = dcg / idcg
        return tkqr
    
    def _calculate_ora(self) -> float:
        """
        Calculate Optimal Round Assessment (ORA).
        
        ORA measures whether a model uses a near optimal number of dialogue rounds.
        It assigns the highest score when n=K, and decreases as n moves away from K.
        
        Returns:
            ORA value in (0, 1]
        """
        import math
        
        n = self.step_count  # Number of rounds in which the interviewer asks questions
        K = self.current_task_total_requirements + 1  # Optimal interaction round count (|Q| + 1)
        
        if K <= 0:
            return 0.0
        
        # Calculate sigma: we want ORA=0.5 when |n-K|=0.5K
        # This yields: sigma = 0.5K / sqrt(2*ln(2)) ≈ 0.425K
        sigma = 0.425 * K
        
        # Calculate ORA using Gaussian-shaped penalty
        # ORA(n, K, σ) = exp(-(n-K)²/(2σ²))
        deviation_squared = (n - K) ** 2
        ora = math.exp(-deviation_squared / (2 * sigma ** 2))
        
        return ora
    
    def _record_step_statistics(self):
        """Record statistics for the current step in the conversation."""
        step_record = {
            "step": self.step_count,
            "total_requirements": self.current_task_total_requirements,
            "total_elicited": self.current_task_elicited,
            "elicitation_ratio": self.current_task_elicited / self.current_task_total_requirements if self.current_task_total_requirements > 0 else 0.0,
        }
        
        self.current_task_step_records.append(step_record)
    
    def _calculate_action_type_effectiveness(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate effectiveness ratio for each action type.
        
        Returns:
            Dictionary mapping action_type to {"total": count, "effective": count, "effectiveness_ratio": float}
        """
        action_effectiveness = {}
        for action_type, stats in self.current_task_action_stats.items():
            total = stats["total"]
            effective = stats["effective"]
            effectiveness_ratio = effective / total if total > 0 else 0.0
            action_effectiveness[action_type] = {
                "total": total,
                "effective": effective,
                "effectiveness_ratio": effectiveness_ratio
            }
        return action_effectiveness
    
    def _calculate_aspect_type_elicitation_ratio(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate elicitation ratio for each aspect type (Interaction, Content, Style).
        
        Returns:
            Dictionary mapping aspect type to {"total": count, "elicited": count, "elicitation_ratio": float}
        """
        aspect_elicitation = {}
        for aspect, stats in self.current_task_requirements_by_aspect.items():
            total = stats["total"]
            elicited = stats["elicited"]
            elicitation_ratio = elicited / total if total > 0 else 0.0
            aspect_elicitation[aspect] = {
                "total": total,
                "elicited": elicited,
                "elicitation_ratio": elicitation_ratio
            }
        return aspect_elicitation
    
    def _record_task_statistics(self):
        """Record statistics for the current completed task."""
        task_id = self.current_task.get("id", f"task_{self.current_task_index - 1}")
        
        # Calculate TKQR and ORA
        tkqr = self._calculate_tkqr()
        ora = self._calculate_ora()
        
        # Calculate action type effectiveness
        action_effectiveness = self._calculate_action_type_effectiveness()
        
        # Calculate aspect type elicitation ratios
        aspect_elicitation = self._calculate_aspect_type_elicitation_ratio()
        
        task_stats = {
            "task_id": task_id,
            "application_type": self.current_task.get("application_type", "Unknown"),  # Save application_type for grouping
            "total_requirements": self.current_task_total_requirements,
            "total_elicited": self.current_task_elicited,
            "elicitation_ratio": self.current_task_elicited / self.current_task_total_requirements if self.current_task_total_requirements > 0 else 0.0,
            "tkqr": tkqr,
            "ora": ora,
            "num_rounds": self.step_count,  # Number of dialogue rounds
            "optimal_rounds": self.current_task_total_requirements + 1,  # K = |Q| + 1
            "token_cost": self.current_task_token_cost,  # Total tokens used for generating questions
            "action_type_effectiveness": action_effectiveness,  # Effectiveness by action type
            "aspect_type_elicitation": aspect_elicitation,  # Elicitation ratio by aspect type
            "step_records": self.current_task_step_records.copy(),  # Include step-by-step records
        }
        
        self.global_stats["task_results"].append(task_stats)
        self.global_stats["task_step_records"].append({
            "task_id": task_id,
            "step_records": self.current_task_step_records.copy(),
        })
        # Store conversation turns for this task
        self.global_stats["conversation_turns"].append({
            "task_id": task_id,
            "task_name": self.current_task.get("name", ""),
            "initial_requirements": self.current_task.get("initial_requirements", ""),
            "user_stories": self.current_task.get("URL", []),
            "user_answer_quality": self.config.user_answer_quality,
            "interviewer_model": self.interviewer_model_name or "unknown",
            "conversation": self.current_task_conversation_turns.copy(),
            "total_turns": len(self.current_task_conversation_turns),
        })
        
        if self.config.verbose:
            print(f"📊 Task {task_id} Statistics:")
            print(f"   Total Requirements: {self.current_task_total_requirements}")
            print(f"   Total Elicited: {self.current_task_elicited} ({task_stats['elicitation_ratio']:.2%})")
            print(f"   TKQR: {tkqr:.4f}")
            print(f"   ORA: {ora:.4f}")
            print(f"   Rounds: {self.step_count} (Optimal: {task_stats['optimal_rounds']})")
            print(f"   Token Cost: {self.current_task_token_cost}")
            print(f"   Action Type Effectiveness:")
            for action_type, stats in action_effectiveness.items():
                print(f"     {action_type}: {stats['effective']}/{stats['total']} = {stats['effectiveness_ratio']:.2%}")
            print(f"   Aspect Type Elicitation:")
            for aspect, stats in aspect_elicitation.items():
                if stats['total'] > 0:  # Only print if there are requirements of this type
                    print(f"     {aspect}: {stats['elicited']}/{stats['total']} = {stats['elicitation_ratio']:.2%}")
            print(f"   Total Steps: {len(self.current_task_step_records)}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics for the current conversation."""
        total_elicited = len(self.elicited_requirements)
        total_requirements = len(self.remaining_requirements) + total_elicited
        
        return {
            "total_requirements": total_requirements,
            "elicited_requirements": total_elicited,
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "remaining_requirements": len(self.remaining_requirements),
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Any]:
        """
        Calculate overall evaluation metrics across all completed tasks.
        
        Returns:
            Dictionary containing:
            - elicitation_ratio: Average ratio of elicited requirements
            - tkqr: Average Turn-discounted Key Question Rate
            - ora: Average Optimal Round Assessment
            - action_type_effectiveness: Overall effectiveness by action type
            - aspect_type_elicitation: Overall elicitation ratio by aspect type (Interaction, Content, Style)
            - total_tasks: Number of tasks evaluated
            - task_results: List of individual task statistics
        """
        if not self.global_stats["task_results"]:
            return {
                "elicitation_ratio": 0.0,
                "tkqr": 0.0,
                "ora": 0.0,
                "action_type_effectiveness": {},
                "aspect_type_elicitation": {},
                "total_tasks": 0,
                "task_results": [],
            }
        
        # Calculate average ratios across all tasks
        total_tasks = len(self.global_stats["task_results"])
        elicitation_ratios = [task["elicitation_ratio"] for task in self.global_stats["task_results"]]
        tkqr_values = [task.get("tkqr", 0.0) for task in self.global_stats["task_results"]]
        ora_values = [task.get("ora", 0.0) for task in self.global_stats["task_results"]]
        token_costs = [task.get("token_cost", 0) for task in self.global_stats["task_results"]]
        
        # Calculate averages
        avg_elicitation_ratio = sum(elicitation_ratios) / total_tasks if total_tasks > 0 else 0.0
        avg_tkqr = sum(tkqr_values) / total_tasks if total_tasks > 0 else 0.0
        avg_ora = sum(ora_values) / total_tasks if total_tasks > 0 else 0.0
        avg_token_cost = sum(token_costs) / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate variances
        def calculate_variance(values, mean):
            """Calculate variance of a list of values."""
            if len(values) <= 1:
                return 0.0
            return sum((x - mean) ** 2 for x in values) / len(values)
        
        variance_elicitation_ratio = calculate_variance(elicitation_ratios, avg_elicitation_ratio)
        variance_tkqr = calculate_variance(tkqr_values, avg_tkqr)
        variance_ora = calculate_variance(ora_values, avg_ora)
        variance_token_cost = calculate_variance(token_costs, avg_token_cost)
        
        # Aggregate action type effectiveness across all tasks
        overall_action_stats = {}  # {action_type: {"total": sum, "effective": sum}}
        for task in self.global_stats["task_results"]:
            action_effectiveness = task.get("action_type_effectiveness", {})
            for action_type, stats in action_effectiveness.items():
                if action_type not in overall_action_stats:
                    overall_action_stats[action_type] = {"total": 0, "effective": 0}
                overall_action_stats[action_type]["total"] += stats["total"]
                overall_action_stats[action_type]["effective"] += stats["effective"]
        
        # Calculate overall effectiveness ratios for each action type
        overall_action_effectiveness = {}
        for action_type, stats in overall_action_stats.items():
            total = stats["total"]
            effective = stats["effective"]
            effectiveness_ratio = effective / total if total > 0 else 0.0
            overall_action_effectiveness[action_type] = {
                "total": total,
                "effective": effective,
                "effectiveness_ratio": effectiveness_ratio
            }
        
        # Aggregate aspect type elicitation across all tasks
        overall_aspect_stats = {}  # {aspect: {"total": sum, "elicited": sum}}
        for task in self.global_stats["task_results"]:
            aspect_elicitation = task.get("aspect_type_elicitation", {})
            for aspect, stats in aspect_elicitation.items():
                if aspect not in overall_aspect_stats:
                    overall_aspect_stats[aspect] = {"total": 0, "elicited": 0}
                overall_aspect_stats[aspect]["total"] += stats["total"]
                overall_aspect_stats[aspect]["elicited"] += stats["elicited"]
        
        # Calculate overall elicitation ratios for each aspect type
        overall_aspect_elicitation = {}
        for aspect, stats in overall_aspect_stats.items():
            total = stats["total"]
            elicited = stats["elicited"]
            elicitation_ratio = elicited / total if total > 0 else 0.0
            overall_aspect_elicitation[aspect] = {
                "total": total,
                "elicited": elicited,
                "elicitation_ratio": elicitation_ratio
            }
        
        # Calculate total counts across all tasks
        total_requirements_all = sum(task["total_requirements"] for task in self.global_stats["task_results"])
        total_elicited_all = sum(task["total_elicited"] for task in self.global_stats["task_results"])
        
        # Calculate overall ratios based on totals
        elicitation_ratio_from_totals = total_elicited_all / total_requirements_all if total_requirements_all > 0 else 0.0
        
        # Group tasks by application_type and calculate statistics
        application_type_stats = {}  # {application_type: {tasks: [], metrics: {...}}}
        for task in self.global_stats["task_results"]:
            app_type = task.get("application_type", "Unknown")
            if app_type not in application_type_stats:
                application_type_stats[app_type] = {
                    "tasks": [],
                    "elicitation_ratios": [],
                    "tkqr_values": [],
                    "ora_values": [],
                }
            application_type_stats[app_type]["tasks"].append(task)
            application_type_stats[app_type]["elicitation_ratios"].append(task.get("elicitation_ratio", 0.0))
            application_type_stats[app_type]["tkqr_values"].append(task.get("tkqr", 0.0))
            application_type_stats[app_type]["ora_values"].append(task.get("ora", 0.0))
        
        # Calculate statistics for each application type
        application_type_results = {}
        for app_type, stats in application_type_stats.items():
            num_tasks = len(stats["tasks"])
            if num_tasks == 0:
                continue
            
            # Calculate averages
            avg_er = sum(stats["elicitation_ratios"]) / num_tasks
            avg_tkqr = sum(stats["tkqr_values"]) / num_tasks
            avg_ora = sum(stats["ora_values"]) / num_tasks
            
            # Calculate variances
            var_er = calculate_variance(stats["elicitation_ratios"], avg_er)
            var_tkqr = calculate_variance(stats["tkqr_values"], avg_tkqr)
            var_ora = calculate_variance(stats["ora_values"], avg_ora)
            
            application_type_results[app_type] = {
                "num_tasks": num_tasks,
                "average_elicitation_ratio": avg_er,
                "variance_elicitation_ratio": var_er,
                "average_tkqr": avg_tkqr,
                "variance_tkqr": var_tkqr,
                "average_ora": avg_ora,
                "variance_ora": var_ora,
            }
        
        return {
            # Average ratios (average of per-task ratios)
            "elicitation_ratio": avg_elicitation_ratio,
            "tkqr": avg_tkqr,
            "ora": avg_ora,
            # Variances
            "variance_elicitation_ratio": variance_elicitation_ratio,
            "variance_tkqr": variance_tkqr,
            "variance_ora": variance_ora,
            # Token cost statistics
            "average_token_cost": avg_token_cost,
            "variance_token_cost": variance_token_cost,
            # Overall ratio (based on total counts)
            "elicitation_ratio_from_totals": elicitation_ratio_from_totals,
            # Action type effectiveness (aggregated across all tasks)
            "action_type_effectiveness": overall_action_effectiveness,
            # Aspect type elicitation (aggregated across all tasks)
            "aspect_type_elicitation": overall_aspect_elicitation,
            # Statistics by application type
            "application_type_statistics": application_type_results,
            # Total counts
            "total_tasks": total_tasks,
            "total_requirements_all_tasks": total_requirements_all,
            "total_elicited_all_tasks": total_elicited_all,
            # Individual task results
            "task_results": self.global_stats["task_results"],
        }

    def get_step_records(self, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get step-by-step records for a specific task or all tasks.
        
        Args:
            task_id: If provided, return records for that specific task. 
                     If None, return all task records.
        
        Returns:
            List of step records. If task_id is provided, returns records for that task.
            If task_id is None, returns all task records with task_id included.
        """
        if task_id is not None:
            # Find records for specific task
            for task_record in self.global_stats["task_step_records"]:
                if task_record["task_id"] == task_id:
                    return task_record["step_records"]
            return []
        else:
            # Return all task records
            return self.global_stats["task_step_records"]
    
    def save_step_records(self, file_path: str):
        """
        Save step-by-step records to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        records_data = {
            "total_tasks": len(self.global_stats["task_step_records"]),
            "task_records": self.global_stats["task_step_records"],
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(records_data, f, indent=2, ensure_ascii=False)
        
        if self.config.verbose:
            print(f"💾 Step records saved to {file_path}")

    def run_all_tasks(self, interviewer: "Interviewer") -> Dict[str, Any]:
        """
        Run all tasks with the given interviewer and return evaluation results.
        
        Args:
            interviewer: Interviewer instance to evaluate
            
        Returns:
            Dictionary containing:
            - overall_metrics: Overall evaluation metrics
            - conversation_results: List of conversation records for all tasks
        """
        self.interviewer_model_name = interviewer.model_name
        
        total_tasks = len(self.tasks)
        task_num = 1
        
        print("\n" + "="*60)
        print("开始运行所有任务")
        print("="*60)
        
        while True:
            # Reset environment for next task
            try:
                observation, info = self.reset()
            except Exception as e:
                print(f"重置环境失败: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Check if all tasks are completed
            if info.get("all_tasks_completed", False):
                if self.config.verbose:
                    print(f"\n所有任务已完成！")
                break
            
            task_id = info.get("task_id", f"task_{task_num}")
            task_data = self.current_task
            
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"任务 {task_num}/{total_tasks}: {task_id}")
                print(f"{'='*60}")
                print(f"系统名称: {task_data.get('name', 'N/A')}")
                print(f"应用类型: {task_data.get('application_type', 'N/A')}")
                print(f"初始需求: {task_data.get('initial_requirements', 'N/A')[:100]}...")
                print(f"总需求数: {observation.get('total_requirements', 0)}")
                print(f"\n开始对话...\n")
            
            # Run conversation for current task
            step = 0
            while step < self.config.max_steps:
                # Generate interviewer question
                if self.config.verbose:
                    print(f"[轮次 {step + 1}]")
                
                try:
                    interviewer_question, usage_info = interviewer.ask_question(
                        conversation_history=self.get_conversation_history(),
                        return_usage=True
                    )
                    # Track token usage for current task
                    if usage_info:
                        self.current_task_token_cost += usage_info.get("total_tokens", 0)
                except Exception as e:
                    print(f"生成 interviewer 问题失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("结束对话")
                    break
                
                if not interviewer_question:
                    print("无法生成 interviewer 问题。结束对话。")
                    break
                
                # Execute environment step
                try:
                    observation, reward, terminated, truncated, info = self.step(interviewer_question)
                except Exception as e:
                    print(f"执行步骤失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break
                
                action_info = info.get("action_info", {})
                
                if self.config.verbose:
                    print(f"  动作类型: {action_info.get('action_type', 'unknown')}")
                    print(f"  与隐式需求相关: {action_info.get('is_relevant_to_implied_requirements', False)}")
                    relevant_req_id = action_info.get("relevant_implied_requirements_id")
                    if relevant_req_id:
                        print(f"  相关需求 ID: {relevant_req_id}")
                    print(f"  已获取的需求: {action_info.get('elicited_requirements', [])}")
                    print(f"  Interviewer: {interviewer_question[:80]}...")
                    user_response = action_info.get("user_response", "")
                    if user_response:
                        print(f"  User: {user_response[:80]}...")
                    print(f"  观察: 总需求={observation.get('total_requirements', 0)}, "
                          f"剩余={observation.get('remaining_requirements', 0)}, "
                          f"获取比例={observation.get('elicitation_ratio', 0.0):.2%}")
                
                step += 1
                
                if terminated or truncated:
                    if terminated:
                        if self.config.verbose:
                            print(f"\n对话已终止（interviewer 完成）。")
                    else:
                        if self.config.verbose:
                            print(f"\n对话已截断（达到最大步数: {self.config.max_steps}）。")
                    break
            
            if self.config.verbose:
                print(f"\n任务 {task_num} 完成: 总轮数={len(self.current_task_conversation_turns)}, "
                      f"已获取需求数={len(self.elicited_requirements)}")
            
            task_num += 1
            
            # Safety check to prevent infinite loop
            if task_num > total_tasks:
                if self.config.verbose:
                    print(f"\n已运行所有 {total_tasks} 个任务，停止。")
                break
        
        # Calculate overall evaluation metrics
        if self.config.verbose:
            print("\n" + "="*60)
            print("计算总体评估指标...")
            print("="*60)
        
        overall_metrics = self.evaluate_all_tasks()
        
        if self.config.verbose:
            print(f"\n总体评估结果:")
            print(f"  总测试样本数: {overall_metrics['total_tasks']}")
            print(f"  总隐式需求数: {overall_metrics['total_requirements_all_tasks']}")
            print(f"  总获取数: {overall_metrics['total_elicited_all_tasks']}")
            print(f"\n平均比例（基于测试样本平均）:")
            print(f"  平均获取比例: {overall_metrics['elicitation_ratio']:.2%}")
            print(f"  平均 TKQR: {overall_metrics['tkqr']:.4f}")
            print(f"  平均 ORA: {overall_metrics['ora']:.4f}")
            print(f"\n方差:")
            print(f"  获取比例方差: {overall_metrics.get('variance_elicitation_ratio', 0.0):.6f}")
            print(f"  TKQR 方差: {overall_metrics.get('variance_tkqr', 0.0):.6f}")
            print(f"  ORA 方差: {overall_metrics.get('variance_ora', 0.0):.6f}")
            print(f"\nToken消耗:")
            print(f"  平均Token消耗: {overall_metrics.get('average_token_cost', 0.0):.2f}")
            print(f"  Token消耗方差: {overall_metrics.get('variance_token_cost', 0.0):.6f}")
            print(f"\n总体比例（基于总计数）:")
            print(f"  总获取比例: {overall_metrics['elicitation_ratio_from_totals']:.2%}")
            
            # Application type statistics
            if overall_metrics.get('application_type_statistics'):
                print(f"\n按应用类型统计:")
                for app_type, stats in overall_metrics['application_type_statistics'].items():
                    print(f"  {app_type}:")
                    print(f"    任务数: {stats['num_tasks']}")
                    print(f"    平均获取比例: {stats['average_elicitation_ratio']:.2%} (方差: {stats['variance_elicitation_ratio']:.6f})")
                    print(f"    平均 TKQR: {stats['average_tkqr']:.4f} (方差: {stats['variance_tkqr']:.6f})")
                    print(f"    平均 ORA: {stats['average_ora']:.4f} (方差: {stats['variance_ora']:.6f})")
            
            # Action type effectiveness
            if overall_metrics.get('action_type_effectiveness'):
                print(f"\n动作类型有效性:")
                for action_type, stats in overall_metrics['action_type_effectiveness'].items():
                    print(f"  {action_type}: {stats['effective']}/{stats['total']} = {stats['effectiveness_ratio']:.2%}")
            
            # Aspect type elicitation
            if overall_metrics.get('aspect_type_elicitation'):
                print(f"\n方面类型获取比例:")
                for aspect, stats in overall_metrics['aspect_type_elicitation'].items():
                    if stats['total'] > 0:
                        print(f"  {aspect}: {stats['elicited']}/{stats['total']} = {stats['elicitation_ratio']:.2%}")
        
        return {
            "overall_metrics": overall_metrics,
            "conversation_results": self.global_stats["conversation_turns"],
        }
    
    def save_evaluation_results(self, file_path: Optional[str] = None, interviewer_model_name: str = None):
        """
        Save evaluation results to a JSON file.
        
        Args:
            file_path: Path to save the evaluation results JSON file. 
                      If None, uses self.config.evaluation_result_path.
            interviewer_model_name: Interviewer model name (if not set, uses self.interviewer_model_name)
        """
        # Use config path if file_path is not provided
        if file_path is None:
            if self.config.evaluation_result_path is None:
                raise ValueError("file_path must be provided or set config.evaluation_result_path")
            file_path = self.config.evaluation_result_path
        overall_metrics = self.evaluate_all_tasks()
        
        if not overall_metrics or overall_metrics.get("total_tasks", 0) == 0:
            if self.config.verbose:
                print("警告: 没有评估结果可保存")
            return
        
        interviewer_model = interviewer_model_name or self.interviewer_model_name or "unknown"
        
        # Prepare task results
        task_results = []
        for task_stats in overall_metrics.get('task_results', []):
            task_results.append({
                "task_id": task_stats.get("task_id", ""),
                "total_requirements": task_stats.get("total_requirements", 0),
                "total_elicited": task_stats.get("total_elicited", 0),
                "elicitation_ratio": task_stats.get("elicitation_ratio", 0.0),
                "tkqr": task_stats.get("tkqr", 0.0),
                "ora": task_stats.get("ora", 0.0),
                "num_rounds": task_stats.get("num_rounds", 0),
                "optimal_rounds": task_stats.get("optimal_rounds", 0),
                "token_cost": task_stats.get("token_cost", 0),
                "action_type_effectiveness": task_stats.get("action_type_effectiveness", {}),
                "aspect_type_elicitation": task_stats.get("aspect_type_elicitation", {}),
            })
        
        evaluation_data = {
            "data_path": self.config.data_path,
            "config": {
                "interviewer_model": interviewer_model,
                "judge_model": self.config.judge_model_name,
                "user_model": self.config.user_model_name,
                "user_answer_quality": self.config.user_answer_quality,
                "max_steps": self.config.max_steps,
            },
            "overall_evaluation": {
                "total_test_samples": overall_metrics['total_tasks'],
                "total_hidden_requirements": overall_metrics['total_requirements_all_tasks'],
                "total_elicited": overall_metrics['total_elicited_all_tasks'],
                # Average ratios (average of per-task ratios)
                "average_elicitation_ratio": overall_metrics['elicitation_ratio'],
                "average_tkqr": overall_metrics['tkqr'],
                "average_ora": overall_metrics['ora'],
                # Variances
                "variance_elicitation_ratio": overall_metrics.get('variance_elicitation_ratio', 0.0),
                "variance_tkqr": overall_metrics.get('variance_tkqr', 0.0),
                "variance_ora": overall_metrics.get('variance_ora', 0.0),
                # Token cost statistics
                "average_token_cost": overall_metrics.get('average_token_cost', 0.0),
                "variance_token_cost": overall_metrics.get('variance_token_cost', 0.0),
                # Overall ratio (based on total counts)
                "elicitation_ratio_from_totals": overall_metrics['elicitation_ratio_from_totals'],
                # Action type effectiveness
                "action_type_effectiveness": overall_metrics.get('action_type_effectiveness', {}),
                # Aspect type elicitation
                "aspect_type_elicitation": overall_metrics.get('aspect_type_elicitation', {}),
                # Statistics by application type
                "application_type_statistics": overall_metrics.get('application_type_statistics', {}),
            },
            "task_results": task_results,
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        
        if self.config.verbose:
            print(f"\n评估结果已保存到: {file_path}")
    
    def save_conversation_results(self, file_path: Optional[str] = None):
        """
        Save conversation results to a JSON file.
        
        Args:
            file_path: Path to save the conversation results JSON file.
                      If None, uses self.config.conversation_result_path.
        """
        # Use config path if file_path is not provided
        if file_path is None:
            if self.config.conversation_result_path is None:
                raise ValueError("file_path must be provided or set config.conversation_result_path")
            file_path = self.config.conversation_result_path
        conversation_results = self.global_stats.get("conversation_turns", [])
        
        if not conversation_results:
            if self.config.verbose:
                print("警告: 没有对话记录可保存")
            return
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_results, f, ensure_ascii=False, indent=2)
        
        if self.config.verbose:
            print(f"对话过程已保存到: {file_path}")
            print(f"  包含 {len(conversation_results)} 个任务的对话记录")
    
    def close(self):
        """Clean up the environment."""
        pass
