
import time
from openai import OpenAI
from typing import Dict, Any, List, Tuple, Optional
from .utils import build_history_into_prompt, parse_output_as_json
from .utils import JUDGE_PROMPT_SYSTEM, JUDGE_PROMPT_USER
from .utils import PASSIVE_RESPONSE_SYSTEM, PASSIVE_RESPONSE_USER


def model_call(
    system_prompt: str,
    user_prompt: str,
    model_config: Dict[str, Any],
    return_json: bool = True,
    return_usage: bool = False
) -> Any:
    """
    Make a synchronous model call and optionally parse JSON response.
    
    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model_config: Model configuration dictionary
        return_json: If True, parse and return JSON. If False, return raw text.
        return_usage: If True, return a tuple of (response, usage_info). If False, return only response.
        
    Returns:
        If return_usage=False:
            Parsed JSON response (if return_json=True) or raw text (if return_json=False)
        If return_usage=True:
            Tuple of (response, usage_info) where usage_info is a dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
    """
    # 支持可选的 base_url，用于不同代理/网关
    if "base_url" in model_config and model_config["base_url"]:
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    else:
        client = OpenAI(api_key=model_config["api_key"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try_time = 0
    while try_time < 3:
        try:
            response = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=model_config["timeout"],
                # extra_body={"enable_thinking": False} # Qwen3-235B-A22B-Thinking-2507的设置, 不传默认思考
                # extra_body={"thinking": {"type": "disabled"}} # kimi2.5的设置
            )
            response_text = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage_info = None
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            else:
                usage_info = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            if return_json:
                response_json = parse_output_as_json(response_text)
                if return_usage:
                    return response_json, usage_info
                else:
                    return response_json
            else:
                if return_usage:
                    return response_text, usage_info
                else:
                    return response_text
        except Exception as e:
            print(f"[ReqElicitGym - Model Call] Error calling model: {e}")
            try_time += 1
            if try_time >= 3:
                if return_usage:
                    return ({}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if return_json else ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                else:
                    return {} if return_json else ""
            time.sleep(2)
    if return_usage:
        return ({}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if return_json else ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return {} if return_json else ""


def model_call_with_thinking(
    system_prompt: str,
    user_prompt: str,
    model_config: Dict[str, Any],
    return_json: bool = True,
    return_usage: bool = False
) -> Any:
    """
    Make a synchronous model call and optionally parse JSON response.
    
    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model_config: Model configuration dictionary
        return_json: If True, parse and return JSON. If False, return raw text.
        return_usage: If True, return a tuple of (response, usage_info). If False, return only response.
        
    Returns:
        If return_usage=False:
            Parsed JSON response (if return_json=True) or raw text (if return_json=False)
        If return_usage=True:
            Tuple of (response, usage_info) where usage_info is a dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
    """
    # 支持可选的 base_url，用于不同代理/网关
    if "base_url" in model_config and model_config["base_url"]:
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
    else:
        client = OpenAI(api_key=model_config["api_key"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try_time = 0
    while try_time < 3:
        try:
            response = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                timeout=model_config["timeout"],
                # extra_body={"enable_thinking": True} # glm-4.7的设置
                # extra_body={"thinking": {"type": "enabled"}} # deepseek-v3.2的设置
                # kimi2.5默认开启thinking, 这里不需要设置
                extra_body={"enable_thinking": True} # Qwen3-235B-A22B-Instruct-2507的设置
            )
            response_text = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage_info = None
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            else:
                usage_info = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            if return_json:
                response_json = parse_output_as_json(response_text)
                if return_usage:
                    return response_json, usage_info
                else:
                    return response_json
            else:
                if return_usage:
                    return response_text, usage_info
                else:
                    return response_text
        except Exception as e:
            print(f"[ReqElicitGym - Model Call] Error calling model: {e}")
            try_time += 1
            if try_time >= 3:
                if return_usage:
                    return ({}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if return_json else ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                else:
                    return {} if return_json else ""
            time.sleep(2)
    if return_usage:
        return ({}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}) if return_json else ("", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return {} if return_json else ""


def judge_interviewer_action(
    action: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    remaining_requirements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Judge the type of interviewer action and its relevance to URL.
    
    Args:
        action: The interviewer's question/action (string)
        task: The task description (dictionary)
        model_config: Configuration for the model
        conversation_history: The conversation history (list of dictionaries)
        remaining_requirements: The remaining requirements (list of dictionaries)
    """
    conversation_history_str = build_history_into_prompt(conversation_history, with_note=True)
    initial_requirements_str = task.get("initial_requirements", "")
    
    # Format remaining requirements (适配 v7 格式: Aspect, RequirementText，没有 Corresponding User Story)
    remaining_requirements_str = ""
    for requirement in remaining_requirements:
        req_id = requirement.get("id", "")
        aspect = requirement.get("aspect", "")
        req_text = requirement.get("requirement", "")
        remaining_requirements_str += f"Requirement ID: {req_id}\tAspect: {aspect}\tRequirement: {req_text}\n\n"
    remaining_requirements_str = remaining_requirements_str.strip()

    system_prompt = JUDGE_PROMPT_SYSTEM
    user_prompt = JUDGE_PROMPT_USER.format(
        initial_requirements=initial_requirements_str,
        conversation_history=conversation_history_str,
        remaining_requirements=remaining_requirements_str if remaining_requirements_str else "No remaining requirements.",
        latest_utterance=action
    )
    response_json = model_call(system_prompt, user_prompt, model_config)
    return response_json

def generate_user_response(
    action: str,
    action_judgement: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    simulator_model_config: Dict[str, Any],
    remaining_requirements: List[Dict[str, Any]],
) -> str:
    """
    Generate a simulated user response to the interviewer's action.
    
    Args:
        action: The interviewer's question/action (string)
        action_judgement: The judgement of the interviewer's action (dictionary)
        conversation_history: The conversation history (list of dictionaries)
        simulator_model_config: Configuration for the simulator model

    Returns:
        The simulated user response (string)
    """
    action_type = action_judgement.get("action_type", "probe")
    is_relevant = action_judgement.get("is_relevant_to_implied_requirements", False)
    relevant_req_id = action_judgement.get("relevant_implied_requirements_id")
    
    implied_requirement = None
    if is_relevant and relevant_req_id:
        for req in remaining_requirements:
            if req.get("id") == relevant_req_id:
                implied_requirement = req.get("requirement", "")
                break

    conversation_history_str = build_history_into_prompt(conversation_history, with_note=False)

    system_prompt = PASSIVE_RESPONSE_SYSTEM
    user_prompt = PASSIVE_RESPONSE_USER.format(
        conversation_history=conversation_history_str,
        latest_utterance=action,
        action_type=action_type,
        is_relevant=is_relevant,
        relevant_requirement=implied_requirement if implied_requirement else "null"
    )
    response_json = model_call(system_prompt, user_prompt, simulator_model_config)
    return response_json.get("response", "")


def evaluate_action(
    action: str,
    task: Dict[str, Any],
    judge_model_config: Dict[str, Any],
    user_simulator_config: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    remaining_requirements: List[Dict[str, Any]],
    user_quality_level: str,
) -> Tuple[str, List[str], float, Dict[str, Any]]:
    """
    Evaluate interviewer action and generate user response.
    
    Args:
        action: The interviewer's question/action (string)
        task: The task description (dictionary)
        judge_model_config: Configuration for the judge model
        user_simulator_config: Configuration for the user simulator model
        conversation_history: The conversation history (list of dictionaries)
        remaining_requirements: The remaining requirements (list of dictionaries)
        user_quality_level: User answer quality level
        
    Returns:
        Tuple of (user_response, elicited_requirements, reward, judgement)
    """
    #format: {action_type: str, is_relevant_to_implied_requirements: bool, relevant_implied_requirements_id: str, reasoning: str}
    judgement = judge_interviewer_action(
        action=action,
        task=task,
        model_config=judge_model_config,
        conversation_history=conversation_history,
        remaining_requirements=remaining_requirements,
    )

    if judgement.get("action_type") == "finish":
        return "", [], 0.0, judgement

    simulated_user_response = generate_user_response(
        action=action,
        action_judgement=judgement,
        conversation_history=conversation_history,
        simulator_model_config=user_simulator_config,
        remaining_requirements=remaining_requirements,
    )
    
    # Determine elicited requirements
    is_relevant = judgement.get("is_relevant_to_implied_requirements")
    relevant_req_id = judgement.get("relevant_implied_requirements_id")
    elicited_requirements = []
    if is_relevant and relevant_req_id:
        # check the requirement is in the remaining requirements
        for req in remaining_requirements:
            if req.get("id") == relevant_req_id:
                elicited_requirements.append(relevant_req_id)
                break
    ## TODO: implement reward calculation
    reward = 0.0

    return simulated_user_response, elicited_requirements, reward, judgement


# def generate_proactive_user_response(
#     action: str,
#     requirement: Dict[str, Any],
#     conversation_history: List[Dict[str, str]],
#     simulator_model_config: Dict[str, Any],
# ) -> str:
#     """
#     Generate a proactive user response that actively reveals a hidden requirement.
    
#     Args:
#         action: The interviewer's latest utterance
#         requirement: The requirement to proactively reveal
#         conversation_history: The conversation history
#         simulator_model_config: Configuration for the simulator model
        
#     Returns:
#         The proactive user response (string)
#     """
#     conversation_history_str = build_history_into_prompt(conversation_history, with_note=False)
    
#     requirement_id = requirement.get("id", "")
#     aspect = requirement.get("aspect", "")
#     requirement_text = requirement.get("requirement", "")
    
#     system_prompt = RESPONSE_ELICIT_SYSTEM
#     user_prompt = RESPONSE_ELICIT_USER.format(
#         requirement_id=requirement_id,
#         aspect=aspect,
#         requirement=requirement_text,
#         conversation_history=conversation_history_str,
#         latest_utterance=action
#     )
#     response_json = model_call(system_prompt, user_prompt, simulator_model_config)
#     return response_json.get("response", "")
