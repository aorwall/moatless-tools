"""Tests for the completion parser utilities."""

import pytest

from moatless.flow.schema import CompletionDTO, CompletionInputMessage, CompletionOutput, ToolCall
from moatless.utils.completion_parser import (
    parse_completion, 
    parse_anthropic_response, 
    parse_openai_response, 
    parse_input
)


def test_parse_anthropic_response():
    """Test parsing an Anthropic response."""
    # Sample Anthropic response with text content
    anthropic_text_response = {
        "content": [
            {
                "type": "text",
                "text": "This is a text response."
            }
        ]
    }
    
    # Test with text content only
    completion_dto = CompletionDTO()
    parse_anthropic_response(completion_dto, anthropic_text_response)
    
    assert completion_dto.output is not None
    assert completion_dto.output.content == "This is a text response."
    assert completion_dto.output.tool_calls is None
    
    # Sample Anthropic response with tool use
    anthropic_tool_response = {
        "content": [
            {
                "type": "text",
                "text": "I'll help you with that."
            },
            {
                "type": "tool_use",
                "name": "search",
                "input": {
                    "query": "latest news"
                }
            }
        ]
    }
    
    # Test with text and tool use
    completion_dto = CompletionDTO()
    parse_anthropic_response(completion_dto, anthropic_tool_response)
    
    assert completion_dto.output is not None
    assert completion_dto.output.content == "I'll help you with that."
    assert completion_dto.output.tool_calls is not None
    assert len(completion_dto.output.tool_calls) == 1
    assert completion_dto.output.tool_calls[0].name == "search"
    assert completion_dto.output.tool_calls[0].arguments == {"query": "latest news"}


def test_parse_openai_response():
    """Test parsing an OpenAI response."""
    # Sample OpenAI response with text content
    openai_text_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a text response."
                }
            }
        ]
    }
    
    # Test with text content only
    completion_dto = CompletionDTO()
    parse_openai_response(completion_dto, openai_text_response)
    
    assert completion_dto.output is not None
    assert completion_dto.output.content == "This is a text response."
    assert completion_dto.output.tool_calls is None
    
    # Sample OpenAI response with tool calls
    openai_tool_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll help you with that.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "latest news"}'
                            }
                        }
                    ]
                }
            }
        ]
    }
    
    # Test with text and tool calls
    completion_dto = CompletionDTO()
    parse_openai_response(completion_dto, openai_tool_response)
    
    assert completion_dto.output is not None
    assert completion_dto.output.content == "I'll help you with that."
    assert completion_dto.output.tool_calls is not None
    assert len(completion_dto.output.tool_calls) == 1
    assert completion_dto.output.tool_calls[0].name == "search"
    assert completion_dto.output.tool_calls[0].arguments == {"query": "latest news"}


def test_parse_input():
    """Test parsing input data."""
    # Sample Anthropic-style input
    anthropic_input = {
        "system": "You are a helpful assistant.",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather?"
            }
        ]
    }
    
    # Test Anthropic-style input
    completion_dto = CompletionDTO()
    parse_input(completion_dto, anthropic_input)
    
    assert completion_dto.system_prompt == "You are a helpful assistant."
    assert completion_dto.input is not None
    assert completion_dto.input.role == "user"
    assert completion_dto.input.content == "What's the weather?"
    
    # Sample OpenAI-style input
    openai_input = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What's the weather?"
            }
        ]
    }
    
    # Test OpenAI-style input
    completion_dto = CompletionDTO()
    parse_input(completion_dto, openai_input)
    
    assert completion_dto.system_prompt == "You are a helpful assistant."
    assert completion_dto.input is not None
    assert completion_dto.input.role == "user"
    assert completion_dto.input.content == "What's the weather?"


def test_parse_completion():
    """Test parsing a complete completion."""
    # Sample completion data with Anthropic format
    anthropic_completion = {
        "original_input": {
            "system": "You are a helpful assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather?"
                }
            ]
        },
        "original_response": {
            "content": [
                {
                    "type": "text",
                    "text": "I don't have access to real-time weather data."
                }
            ]
        }
    }
    
    # Test with Anthropic format
    completion_dto = parse_completion(anthropic_completion)
    
    assert completion_dto.original_input == anthropic_completion["original_input"]
    assert completion_dto.original_output == anthropic_completion["original_response"]
    assert completion_dto.system_prompt == "You are a helpful assistant."
    assert completion_dto.input is not None
    assert completion_dto.input.content == "What's the weather?"
    assert completion_dto.output is not None
    assert completion_dto.output.content == "I don't have access to real-time weather data."
    
    # Sample completion data with OpenAI format
    openai_completion = {
        "original_input": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What's the weather?"
                }
            ]
        },
        "original_response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I don't have access to real-time weather data."
                    }
                }
            ]
        }
    }
    
    # Test with OpenAI format
    completion_dto = parse_completion(openai_completion)
    
    assert completion_dto.original_input == openai_completion["original_input"]
    assert completion_dto.original_output == openai_completion["original_response"]
    assert completion_dto.system_prompt == "You are a helpful assistant."
    assert completion_dto.input is not None
    assert completion_dto.input.content == "What's the weather?"
    assert completion_dto.output is not None
    assert completion_dto.output.content == "I don't have access to real-time weather data."


def test_parse_completion_error_handling():
    """Test error handling in parse_completion."""
    # Malformed input
    malformed_completion = {
        "original_input": {"invalid": "format"},
        "original_response": {"invalid": "format"}
    }
    
    # Parsing should not raise an exception but return a partially populated DTO
    completion_dto = parse_completion(malformed_completion)
    
    assert completion_dto.original_input == malformed_completion["original_input"]
    assert completion_dto.original_output == malformed_completion["original_response"]
    assert completion_dto.system_prompt is None
    
    # Missing fields should be handled gracefully
    incomplete_completion = {}
    completion_dto = parse_completion(incomplete_completion)
    
    assert completion_dto.original_input is None
    assert completion_dto.original_output is None


def test_parse_anthropic_new_format():
    """Test parsing Anthropic's newer format with system at top level and content arrays."""
    # Sample Anthropic new format input
    anthropic_new_format = {
        "model": "claude-3-7-sonnet-20250219",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in San Francisco?"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I don't have access to real-time weather data."
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "search",
                        "input": {
                            "query": "weather in San Francisco"
                        }
                    }
                ]
            }
        ],
        "system": [
            {
                "type": "text",
                "text": "You are an autonomous AI assistant with superior programming skills."
            }
        ]
    }
    
    # Test parsing the complete input data
    completion_dto = CompletionDTO()
    parse_input(completion_dto, anthropic_new_format)
    
    # Verify the results
    assert completion_dto.system_prompt == "You are an autonomous AI assistant with superior programming skills."
    assert completion_dto.input is not None
    assert completion_dto.input.content == "What's the weather in San Francisco?"
    assert completion_dto.input.role == "user"


def test_parse_anthropic_complete_new_format():
    """Test parsing a complete Anthropic completion with the newer format."""
    # Sample Anthropic new format completion data
    anthropic_new_format_completion = {
        "original_input": {
            "model": "claude-3-7-sonnet-20250219",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's the weather in San Francisco?"
                        }
                    ]
                }
            ],
            "system": [
                {
                    "type": "text", 
                    "text": "You are an autonomous AI assistant with superior programming skills."
                }
            ]
        },
        "original_response": {
            "content": [
                {
                    "type": "text",
                    "text": "I don't have access to real-time weather data."
                },
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "search",
                    "input": {
                        "query": "weather in San Francisco"
                    }
                }
            ]
        }
    }
    
    # Parse the complete completion
    completion_dto = parse_completion(anthropic_new_format_completion)
    
    # Verify the results
    assert completion_dto.original_input == anthropic_new_format_completion["original_input"]
    assert completion_dto.original_output == anthropic_new_format_completion["original_response"]
    assert completion_dto.system_prompt == "You are an autonomous AI assistant with superior programming skills."
    assert completion_dto.input is not None
    assert completion_dto.input.content == "What's the weather in San Francisco?"
    assert completion_dto.output is not None
    assert completion_dto.output.content == "I don't have access to real-time weather data."
    assert completion_dto.output.tool_calls is not None
    assert len(completion_dto.output.tool_calls) == 1
    assert completion_dto.output.tool_calls[0].name == "search"
    assert completion_dto.output.tool_calls[0].arguments == {"query": "weather in San Francisco"}


def test_parse_anthropic_example_format():
    """Test parsing the exact Anthropic format shown in the example."""
    # Sample based on the provided format example
    anthropic_example = {
        "original_input": {
            "model": "claude-3-7-sonnet-20250219",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Solve the following issue:\nMigrations uses value of enum object instead of its name."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I'll help solve this issue with Django migrations using Enum objects as default values."
                        },
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "Think",
                            "input": {
                                "thought": "The issue is about Django migrations when using Enum objects as default values for CharField."
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "The thought was logged"
                                }
                            ]
                        }
                    ]
                }
            ],
            "system": [
                {
                    "type": "text",
                    "text": "You are an autonomous AI assistant with superior programming skills. As you're working autonomously, you cannot communicate with the user but must rely on information you can get from the available functions."
                }
            ]
        },
        "original_response": {
            "content": [
                {
                    "type": "text",
                    "text": "Let me search for code related to Django's migration system and how it handles default values."
                },
                {
                    "type": "tool_use",
                    "id": "tool_2",
                    "name": "SemanticSearch",
                    "input": {
                        "file_pattern": "django/db/migrations/*.py",
                        "query": "django migration serialization of default values enum",
                        "category": "implementation"
                    }
                }
            ]
        }
    }
    
    # Parse the example format
    completion_dto = parse_completion(anthropic_example)
    
    # Verify the results
    assert completion_dto.original_input == anthropic_example["original_input"]
    assert completion_dto.original_output == anthropic_example["original_response"]
    assert completion_dto.system_prompt == "You are an autonomous AI assistant with superior programming skills. As you're working autonomously, you cannot communicate with the user but must rely on information you can get from the available functions."
    
    # For this specific multi-message format, we should get the last user message
    assert completion_dto.input is not None
    assert "tool_result" in completion_dto.input.content
    assert "thought was logged" in completion_dto.input.content
    
    # Check the output message and tool use
    assert completion_dto.output is not None
    assert "Let me search for code" in completion_dto.output.content
    assert completion_dto.output.tool_calls is not None
    assert len(completion_dto.output.tool_calls) == 1
    assert completion_dto.output.tool_calls[0].name == "SemanticSearch" 