"""Evaluation configuration settings"""

from datetime import datetime

# Default evaluation settings
DEFAULT_CONFIG = {
    # Model settings
    "api_key": None,
    "base_url": None,
    # Dataset settings
    "split": "lite_and_verified_solvable",
    "instance_ids": None,
    # Tree search settings
    "max_iterations": 20,
    "max_expansions": 1,
    "max_cost": 1.0,
    # Runner settings
    "num_workers": 10,
    # Evaluation settings
    "evaluation_name": None,
    "rerun_errors": False,
}

# Configuration for deepseek-chat with tool_call format
DEEPSEEK_TOOL_CALL_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
}

# Configuration for deepseek-chat with tool_call format
DEEPSEEK_TOOL_CALL_SUMMARY_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "tool_call",
    "message_history": "summary",
    "thoughts_in_action": False,
}

# Configuration for deepseek-chat with react format
DEEPSEEK_REACT_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "react",
    "message_history": "react",
}

# Configuration for GPT-4o-mini with tool_call format
GPT4O_MINI_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "azure/gpt-4o-mini",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": True,
}

# Configuration for GPT-4o with tool_call format
GPT4O_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "azure/gpt-4o",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": True,
}

# Configuration for GPT-4o with tool_call format
CLAUDE_35_SONNET_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "claude-3-5-sonnet-20241022",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
    "split": "lite_and_verified_solvable",
}


# Configuration for single instance runs
def get_single_instance_config(
    instance_id: str, base_config: dict = DEEPSEEK_TOOL_CALL_CONFIG
) -> dict:
    """Create a configuration for running a single instance"""
    return {
        **base_config,
        "instance_ids": [instance_id],
        "num_workers": 1,  # Override to 1 for single instance
        "evaluation_name": f"single_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


# Example single instance configurations
DJANGO_17051_DEEPSEEK = get_single_instance_config(
    "django__django-17051", DEEPSEEK_REACT_CONFIG
)
DJANGO_17051_GPT4 = get_single_instance_config(
    "django__django-17051", GPT4O_MINI_CONFIG
)

# Active configuration - change this to switch between configs
ACTIVE_CONFIG = DJANGO_17051_DEEPSEEK  # Change this to run different configurations


def get_config() -> dict:
    """Get the active configuration"""
    return ACTIVE_CONFIG
