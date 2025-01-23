"""Model configuration settings for different LLMs."""

from moatless.completion.base import LLMResponseFormat
from moatless.schema import MessageHistoryType

# Base configuration that all models inherit from
BASE_MODEL_CONFIG = {
    "temperature": 0.0,
    "thoughts_in_action": False,
    "max_tokens": 4000,
    "few_shot_examples": True,
}

# Claude 3.5 Sonnet configuration
CLAUDE_35_SONNET = {
    **BASE_MODEL_CONFIG,
    "model": "claude-3-5-sonnet-20241022",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
}

# Claude 3.5 Haiku configuration
CLAUDE_35_HAIKU = {
    **BASE_MODEL_CONFIG,
    "model": "claude-3-5-haiku-20241022",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
}

# GPT-4o configuration
GPT4O = {
    **BASE_MODEL_CONFIG,
    "model": "gpt-4o-2024-11-20",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "thoughts_in_action": True,
}

AZURE_GPT4O = {
    **GPT4O,
    "model": "azure/gpt-4o"
}

# GPT-4o Mini configuration
GPT4O_MINI = {
    **BASE_MODEL_CONFIG,
    "model": "gpt-4o-mini-2024-07-18",
    "response_format": LLMResponseFormat.TOOLS,
    "message_history_type": MessageHistoryType.MESSAGES,
    "thoughts_in_action": True,
}

# O1 Preview configuration
O1_PREVIEW = {
    **BASE_MODEL_CONFIG,
    "model": "o1-preview-2024-09-12",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "max_tokens": 8000,
}

# O1 Mini configuration
O1_MINI = {
    **BASE_MODEL_CONFIG,
    "model": "o1-mini-2024-09-12",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "disable_thoughts": True,
    "max_tokens": 8000,
}

# DeepSeek Chat configuration
DEEPSEEK_CHAT = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# DeepSeek Reasoner Chat configuration
DEEPSEEK_REASONER_CHAT = {
    **BASE_MODEL_CONFIG,
    "model": "deepseek/deepseek-reasoner",
    "temperature": None,
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
    "disable_thoughts": True,
    "few_shot_examples": True,
    "merge_same_role_messages": True,
    "max_tokens": 8000,
}

# Gemini 1206 configuration
GEMINI_1206 = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-exp-1206",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Gemini Flash configuration
GEMINI_FLASH = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-2.0-flash-exp",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Gemini Flash Think configuration
GEMINI_FLASH_THINK = {
    **BASE_MODEL_CONFIG,
    "model": "gemini/gemini-2.0-flash-thinking-exp",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Llama 3.1 405B Instruct configuration
LLAMA_31_405B = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/meta-llama/llama-3.1-405b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Llama 3.1 70B Instruct configuration
LLAMA_31_70B = {
    **BASE_MODEL_CONFIG,
    "model": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

# Qwen 2.5 Coder configuration
QWEN_25_CODER = {
    **BASE_MODEL_CONFIG,
    "model": "qwen/qwen-2.5-coder-32b-instruct",
    "response_format": LLMResponseFormat.REACT,
    "message_history_type": MessageHistoryType.REACT,
}

SUPPORTED_MODELS = [
    #AZURE_GPT4O,
    CLAUDE_35_SONNET,
    CLAUDE_35_HAIKU,
    O1_MINI,
    GPT4O,
    GPT4O_MINI,
    GEMINI_1206,
    GEMINI_FLASH,
    DEEPSEEK_CHAT,
    DEEPSEEK_REASONER_CHAT,
    LLAMA_31_70B,
    QWEN_25_CODER,
]

MODEL_CONFIGS = {config["model"]: config for config in SUPPORTED_MODELS}


def get_model_config(model_name: str) -> dict:
    """Get the configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in supported models")
    return MODEL_CONFIGS[model_name]
