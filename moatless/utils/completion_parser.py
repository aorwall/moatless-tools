"""Utilities for parsing completions from different LLM providers."""

import json
import logging
from typing import Dict, Any

from moatless.flow.schema import CompletionDTO, CompletionInputMessage, CompletionOutput, ToolCall

logger = logging.getLogger(__name__)


def parse_completion(completion_data: Dict[str, Any]) -> CompletionDTO:
    """Parse a completion from raw data into a CompletionDTO object.

    Supports both OpenAI and Anthropic formats.

    Args:
        completion_data: The raw completion data, with original_input and original_response fields

    Returns:
        A CompletionDTO object with parsed data
    """
    # Create a CompletionDTO object to store parsed data
    completion_dto = CompletionDTO(
        original_input=completion_data.get("original_input"), original_output=completion_data.get("original_response")
    )

    # Parse from original_response if it exists
    original_response = completion_data.get("original_response") or completion_data.get("original_response_obj")
    if original_response:
        if isinstance(original_response, str):
            completion_dto.output = CompletionOutput(content=original_response)
        # Handle Anthropic format
        if "content" in original_response:
            completion_dto.output = parse_anthropic_response(completion_dto, original_response)
        # Handle OpenAI format
        elif "choices" in original_response:
            completion_dto.output = parse_openai_response(completion_dto, original_response)

    # Parse original input if available
    original_input = completion_data.get("original_input")
    if original_input:
        parse_input(completion_dto, original_input)
        
    if "traceback_exception" in completion_data:
        completion_dto.error = str(completion_data.get("traceback_exception"))

    return completion_dto


def parse_anthropic_response(completion_dto: CompletionDTO, response: Dict[str, Any]) -> CompletionOutput:
    """Parse Anthropic style response.

    Args:
        completion_dto: The CompletionDTO to update
        response: The Anthropic response data
    """
    output_content = None
    tool_calls = None

    for content_block in response.get("content", []):
        if content_block.get("type") == "text":
            output_content = content_block.get("text")
        elif content_block.get("type") == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append(ToolCall(name=content_block.get("name", ""), arguments=content_block.get("input", {})))

    return CompletionOutput(content=output_content, tool_calls=tool_calls)


def parse_openai_response(completion_dto: CompletionDTO, response: Dict[str, Any]) -> CompletionOutput:
    """Parse OpenAI style response.

    Args:
        completion_dto: The CompletionDTO to update
        response: The OpenAI response data
    """
    if response.get("choices") and len(response["choices"]) > 0:
        choice = response["choices"][0]
        message = choice.get("message", {})

        output_content = message.get("content")
        tool_calls = None

        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                try:
                    function = tool_call.get("function", {})
                    arguments = json.loads(function.get("arguments", "{}"))
                    tool_calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool call arguments: {function.get('arguments')}")

        return CompletionOutput(content=output_content, tool_calls=tool_calls)


def parse_input(completion_dto: CompletionDTO, input_data: Dict[str, Any]) -> None:
    """Parse input data from either Anthropic or OpenAI format.

    Args:
        completion_dto: The CompletionDTO to update
        input_data: The input data
    """
    system_prompt = None
    input_messages = []

    # Extract system prompt - handle different formats
    if "system" in input_data:
        # Handle system as array of content blocks (newer Anthropic format)
        if isinstance(input_data["system"], list):
            for content_block in input_data["system"]:
                if content_block.get("type") == "text":
                    system_prompt = content_block.get("text")
                    break
        # Handle system as direct string (older format)
        else:
            system_prompt = input_data["system"]
    # Look for system message in the messages array
    elif "messages" in input_data:
        for message in input_data["messages"]:
            if message.get("role") == "system":
                # Handle content as array of blocks
                if isinstance(message.get("content"), list):
                    for content_block in message["content"]:
                        if content_block.get("type") == "text":
                            system_prompt = content_block.get("text")
                            break
                # Handle content as direct string
                else:
                    system_prompt = message.get("content")
                break

    # Add system prompt as first message if it exists
    if system_prompt:
        input_messages.append(
            CompletionInputMessage(
                role="system",
                content=system_prompt,
                tool_calls=None,
            )
        )

    # Parse all messages in the input
    if "messages" in input_data:
        for message in input_data["messages"]:
            # Skip system messages as they're handled separately and already added above
            if message.get("role") == "system":
                continue

            content = message.get("content")
            parsed_tool_calls = None

            # Parse tool_calls if they exist
            if "tool_calls" in message and message["tool_calls"]:
                parsed_tool_calls = []
                for tool_call in message["tool_calls"]:
                    # Handle OpenAI format
                    if "function" in tool_call:
                        try:
                            function = tool_call.get("function", {})
                            arguments = {}
                            if isinstance(function.get("arguments"), str):
                                try:
                                    arguments = json.loads(function.get("arguments", "{}"))
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse tool call arguments: {function.get('arguments')}")
                                    arguments = {"raw_arguments": function.get("arguments", "")}
                            else:
                                arguments = function.get("arguments", {})

                            parsed_tool_calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))
                        except Exception as e:
                            logger.warning(f"Error parsing OpenAI tool call: {e}")
                    # Handle Anthropic format
                    elif "name" in tool_call and "input" in tool_call:
                        parsed_tool_calls.append(
                            ToolCall(name=tool_call.get("name", ""), arguments=tool_call.get("input", {}))
                        )

            # Handle content as array of blocks (newer Anthropic format)
            if isinstance(content, list):
                # Process the complex content to create a representative string
                text_content = []
                for content_block in content:
                    # For tool_result blocks, preserve the structure
                    if content_block.get("type") == "tool_result":
                        tool_id = content_block.get("tool_use_id", "")
                        tool_content = []
                        for result_content in content_block.get("content", []):
                            if result_content.get("type") == "text":
                                tool_content.append(result_content.get("text", ""))

                        result_text = f"tool_result (id: {tool_id}): {' '.join(tool_content)}"
                        text_content.append(result_text)
                    # For text blocks, add the text directly
                    elif content_block.get("type") == "text":
                        text_content.append(content_block.get("text", ""))
                    # For tool_use blocks, these are tool calls in the Anthropic format
                    elif content_block.get("type") == "tool_use":
                        if parsed_tool_calls is None:
                            parsed_tool_calls = []
                        parsed_tool_calls.append(
                            ToolCall(name=content_block.get("name", ""), arguments=content_block.get("input", {}))
                        )
                    # For other block types, include their type and basic info
                    else:
                        block_type = content_block.get("type", "unknown")
                        text_content.append(f"{block_type}_block")

                input_messages.append(
                    CompletionInputMessage(
                        role=message.get("role", "user"),
                        content="\n".join(text_content) if text_content else None,
                        tool_calls=parsed_tool_calls,
                    )
                )
            # Handle content as direct string (older format)
            else:
                input_messages.append(
                    CompletionInputMessage(
                        role=message.get("role", "user"),
                        content=content,
                        tool_calls=parsed_tool_calls,
                    )
                )

    # Keep system_prompt for backward compatibility
    completion_dto.system_prompt = system_prompt
    if input_messages:
        completion_dto.input = input_messages
