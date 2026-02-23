import json
import re
from typing import Dict, Any, List, Tuple, Optional

def parse_output_as_json(response_text: str) -> Dict[str, Any]:
    """
    Parse the model's response text and extract JSON from it.
    
    Args:
        response_text: Raw response text from the model
        
    Returns:
        Parsed JSON dictionary, or empty dict if parsing fails
    """
    try:
        # Try to parse the entire response as JSON
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Look for JSON blocks in the response
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                pass
        
        # Look for JSON-like content between braces
        brace_pattern = r'\{.*\}'
        matches = re.findall(brace_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # If all parsing attempts fail, return empty dict
        return {}

def build_history_into_prompt(conversation_history: List[Dict[str, str]], with_note: bool = False) -> str:
    """
    Convert conversation history into a formatted prompt string.
    
    Args:
        conversation_history: List of conversation entries with 'role' and 'content' keys
        with_note: Whether to include notes in the conversation history
        
    Returns:
        Formatted conversation history string
    """
    if not conversation_history:
        return "No previous conversation."
    
    history_str = ""
    for entry in conversation_history:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        note = entry.get("note", "") if with_note else ""
        
        if role == "interviewer":
            history_str += f"Interviewer: {content}\n"
            if with_note and note:
                history_str += f"[Note: {note}]\n"
        elif role == "user":
            history_str += f"User: {content}\n"
        else:
            history_str += f"{role.capitalize()}: {content}\n"
        
        history_str += "\n"
    
    return history_str.strip()

JUDGE_PROMPT_SYSTEM = """## **Task**
You are an expert judge evaluating the type of an interviewer's action in a requirement elicitation conversation.

## **Action Types**
1. **clarify**: The interviewer is asking for clarification about something mentioned in the conversation (e.g., "What do you mean by X?", "Can you explain Y?")
2. **probe**: The interviewer is probing for new information or deeper details about requirements (e.g., "What about Z?", "Do you need feature A?")
3. **finish**: The interviewer is indicating the conversation should end and generating a comprehensive set of user stories as the final output artifact. The finish action typically includes a statement like "I have gathered enough information" followed by a set of user stories in the format "As a [user type], I want to [action/goal] so that [benefit/value]." (e.g., "I have gathered enough information. Based on our conversation, here is the set of user stories:\n\n User Story 1: As a user, I want to ... so that ...\n\n User Story 2: As a user, I want to ... so that ...\n\n ...")

## **Instruction**
1. Analyze the interviewer's latest utterance in the context of the conversation
2. Determine the action type (clarify/probe/finish)
3. If the action is probing, check if it's explicitly asking for requirements that you have, asking for requirements that you don't have, giving a too general query.
4. Provide your assessment in JSON format


## **Example Format**
```json
{
    "action_type": "clarify/probe/finish",
    "is_relevant_to_implied_requirements": true/false,
    "relevant_implied_requirements_id": "requirement_id if relevant, or null",
    "reasoning": "brief explanation"
}
```

## **Important Notes**
- Be precise in identifying action types
- For "probe" actions, check if the question relates to any remaining_requirements
- "is_relevant_to_implied_requirements" should be true only if the probe is asking about something covered in the remaining_requirements
- If action is not "probe", "is_relevant_to_implied_requirements" can be false
"""

JUDGE_PROMPT_USER = """
**Initial Requirements:**
{initial_requirements}

**Conversation History:**
{conversation_history}

**Interviewer's Latest Utterance:**
{latest_utterance}

**Remaining Requirements:**
{remaining_requirements}

Please analyze the interviewer's action and provide your assessment in JSON format wrapped in ```json and ```.
"""

PASSIVE_RESPONSE_SYSTEM = """## **Task**
You are a user being interviewed about your software requirements. Respond naturally to the interviewer's questions or statements.

## **Instructions**
1. Answer based on your implied requirements if the question is relevant
2. If the question is NOT relevant to your requirements, politely say you don't care about that
3. Keep responses BRIEF and CONCISE (1-3 sentences max)
4. Be natural and conversational, like a real person would answer
5. Adjust your response based on the conversation context

Return JSON: {"response": "your response"}
"""


PASSIVE_RESPONSE_USER = """
**Conversation History:**
{conversation_history}

**Interviewer's Latest Utterance:**
{latest_utterance}

Please respond naturally to the interviewer's question or statement.

**Context of this latest utterance:**
- Action type: {action_type}
- Is relevant to your requirements: {is_relevant}
- Relevant Requirement: {relevant_requirement} 
Note: if is_relevant is false, the relevant_requirement is null.
"""
