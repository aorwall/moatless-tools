import enum
import random
import string

import instructor


class LLMResponseFormat(enum.Enum):
    TOOLS = "tool_call"
    ANTHROPIC_TOOLS = "anthropic_tools"
    JSON = "json_mode"
    STRUCTURED_OUTPUT = "structured_output"


def response_format_by_model(model: str) -> LLMResponseFormat | None:
    if "azure" in model:
        return LLMResponseFormat.TOOLS

    if "gpt" in model:
        return LLMResponseFormat.STRUCTURED_OUTPUT

    if model.startswith("claude") or model.startswith("anthropic.claude"):
        return LLMResponseFormat.ANTHROPIC_TOOLS

    if "claude" in model:
        return LLMResponseFormat.TOOLS

    if model.startswith("openrouter/anthropic/claude"):
        return LLMResponseFormat.TOOLS

    return LLMResponseFormat.JSON


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
