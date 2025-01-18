import json
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse, Usage, Message

from moatless.completion.base import LLMResponseFormat, CompletionRetryError
from moatless.completion.json import JsonCompletionModel
from moatless.exceptions import CompletionRejectError


@pytest.fixture
def mock_litellm_json_response():
    """Mock LiteLLM response with JSON content"""
    def _create_mock(content, usage=None):
        # Create message
        message = Message(
            content=content,
            role="assistant"
        )

        # Create usage
        if usage:
            usage_obj = Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            )
        else:
            usage_obj = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        # Create ModelResponse
        return ModelResponse(
            id="test_id",
            created=1234567890,
            model="test",
            choices=[{
                "message": message,
                "finish_reason": "stop",
                "index": 0
            }],
            usage=usage_obj
        )

    return _create_mock


@patch("litellm.completion")
def test_validate_completion_valid_json(mock_completion, mock_litellm_json_response, test_schema, test_messages):
    """Test validating valid JSON response"""
    model = JsonCompletionModel(
        model="test",
        response_format=LLMResponseFormat.JSON,
    )
    model.initialize(test_schema, "test")
    
    # Mock valid JSON response
    valid_json = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing the command"
    }
    mock_response = mock_litellm_json_response(
        json.dumps(valid_json),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )

    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert structured_outputs[0].thoughts == "Testing the command"
    assert text_response is None
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_invalid_json(mock_completion, mock_litellm_json_response, test_schema, test_messages):
    """Test validating invalid JSON response"""
    model = JsonCompletionModel(
        model="test",
        response_format=LLMResponseFormat.JSON,
    )
    model.initialize(test_schema, "test")
    
    # Mock invalid JSON response
    mock_response = mock_litellm_json_response(
        "invalid json",
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    )
    
    with pytest.raises(CompletionRetryError):
        model._validate_completion(
            completion_response=mock_response,
        )


@patch("litellm.completion")
def test_validate_completion_missing_required(mock_completion, mock_litellm_json_response, test_schema, test_messages):
    """Test validating JSON missing required fields"""
    model = JsonCompletionModel(
        model="test",
        response_format=LLMResponseFormat.JSON,
    )
    model.initialize(test_schema, "test")
    
    # Mock JSON missing required field
    invalid_json = {
        "args": ["--flag"]  # Missing required 'command'
    }
    mock_response = mock_litellm_json_response(
        json.dumps(invalid_json),
        usage={"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
    )
    
    with pytest.raises(CompletionRetryError):
        model._validate_completion(
            completion_response=mock_response,
        )


@patch("litellm.completion")
def test_end_to_end_completion(mock_completion, mock_litellm_json_response, test_schema, test_messages):
    """Test complete JSON completion flow"""
    model = JsonCompletionModel(
        model="test",
        response_format=LLMResponseFormat.JSON,
    )
    model.initialize(test_schema, "test")
    
    valid_json = {
        "command": "test",
        "args": ["--flag"],
        "thoughts": "Testing the command"
    }
    mock_response = mock_litellm_json_response(
        json.dumps(valid_json),
        usage={"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
    )
    mock_completion.return_value = mock_response
    
    result = model.create_completion(messages=test_messages)
    print(result)
    
    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"]
    assert result.structured_outputs[0].thoughts == "Testing the command"
    
    # Verify correct parameters were passed
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert len(call_kwargs["messages"]) == 2