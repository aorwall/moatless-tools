from typing import Optional, List, Dict, Any

import litellm
from litellm import token_counter

from moatless.settings import Settings

_trace_metadata = {}


def set_trace_metadata(new_metadata: Dict[str, Any]):
    global _trace_metadata
    _trace_metadata = new_metadata


def update_trace_metadata(key: str, value):
    global _trace_metadata
    _trace_metadata[key] = value


_mock_response = None


def set_mock_response(response):
    global _mock_response
    _mock_response = response


def completion(
    model: str,
    messages: List,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    trace_name: str = "moatless-agent",
    stop: Optional[List[str]] = None,
    generation_name: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> litellm.ModelResponse:
    if len(messages) == 0:
        raise ValueError("At least one message is required.")

    global _trace_metadata, _mock_response
    metadata = {}
    metadata.update(_trace_metadata)

    if generation_name:
        metadata["generation_name"] = generation_name

    metadata["trace_name"] = trace_name

    tokens = token_counter(messages=messages[-1:])
    if tokens > Settings.max_message_tokens:
        raise ValueError(f"Too many tokens in the new message: {tokens}")

    response = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools,
        stop=stop,
        metadata=metadata,
        messages=messages,
        mock_response=_mock_response,
    )

    return response
