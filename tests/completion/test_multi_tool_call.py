import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from litellm.types.utils import ModelResponse, Usage, Message, Function, ChatCompletionMessageToolCall
from moatless.actions.schema import ActionArguments
from moatless.completion.base import CompletionRetryError
from moatless.completion.multi_tool_call import MultiToolCallCompletionModel
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError
from pydantic import Field


class TestActionArgs(ActionArguments):
    command: str = Field(..., description="The command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")

    model_config = {"title": "TestActionArgs"}


class AnotherActionArgs(ActionArguments):
    path: str = Field(..., description="The file path to process")
    recursive: bool = Field(default=False, description="Whether to process recursively")

    model_config = {"title": "AnotherActionArgs"}


@pytest.fixture
def test_schemas():
    return [TestActionArgs, AnotherActionArgs]


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "Run the test command and process the path"}]


@pytest.fixture
def mock_litellm_multi_tool_response():
    """Mock LiteLLM response with a MultiToolCall wrapper containing multiple tool calls"""

    def _create_mock(tool_calls=None, usage=None):
        # Create the MultiToolCall wrapper containing multiple tool calls
        wrapper_args = {
            "tool_calls": tool_calls or [],
            "thoughts": "Processing multiple tools"
        }
        
        # Create a single tool call that contains the wrapper
        function = Function(name="MultiToolCall", arguments=json.dumps(wrapper_args))
        message_tool_calls = [
            ChatCompletionMessageToolCall(
                function=function, id="wrapper-call", type="function"
            )
        ]

        # Create message
        message = Message(content="Test response", tool_calls=message_tool_calls, role="assistant")

        # Create usage
        if usage:
            usage_obj = Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            )
        else:
            usage_obj = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        # Create ModelResponse
        return ModelResponse(
            id="test_id",
            created=1234567890,
            model="test",
            choices=[{"message": message, "finish_reason": "stop", "index": 0}],
            usage=usage_obj,
        )

    return _create_mock


def test_prepare_schema_multiple(test_schemas):
    """Test preparing multiple tool schemas"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")
    assert len(model._response_schema) == 2
    assert model._response_schema[0] == test_schemas[0]
    assert model._response_schema[1] == test_schemas[1]


def test_get_completion_params(test_schemas):
    """Test getting tool completion parameters with wrapper"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    assert model._completion_params
    assert "tools" in model._completion_params
    assert len(model._completion_params["tools"]) == 3  # The original tools plus the wrapper
    assert model._completion_params["tools"][2]["type"] == "function"
    assert model._completion_params["tools"][2]["function"]["name"] == "MultiToolCall"
    assert "tool_choice" in model._completion_params


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_multiple_tools(
    mock_completion, mock_litellm_multi_tool_response, test_schemas, test_messages
):
    """Test validating response with multiple tool calls in wrapper"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # Mock multiple tool calls in the wrapper
    tool_calls = [
        {
            "name": "TestActionArgs",
            "args": {"command": "test", "args": ["--flag"], "thoughts": "Testing command execution"}
        },
        {
            "name": "AnotherActionArgs",
            "args": {"path": "/tmp/test", "recursive": True, "thoughts": "Testing path processing"}
        }
    ]
    
    mock_response = mock_litellm_multi_tool_response(tool_calls=tool_calls)

    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)

    assert structured_outputs
    assert len(structured_outputs) == 2
    
    # Verify first tool call
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert structured_outputs[0].thoughts == "Testing command execution"
    
    # Verify second tool call
    assert structured_outputs[1].path == "/tmp/test"
    assert structured_outputs[1].recursive is True
    assert structured_outputs[1].thoughts == "Testing path processing"
    
    assert text_response == "Test response"
    assert thought == "Processing multiple tools"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_partial_valid_tools(
    mock_completion, mock_litellm_multi_tool_response, test_schemas, test_messages
):
    """Test validating response with mix of valid and invalid tool calls"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # Mix of valid and invalid tool calls
    tool_calls = [
        {
            "name": "TestActionArgs",
            "args": {"command": "test", "args": ["--flag"], "thoughts": "Testing command execution"}
        },
        {
            "name": "AnotherActionArgs",
            "args": {"recursive": True}  # Missing required 'path' field
        }
    ]
    
    mock_response = mock_litellm_multi_tool_response(tool_calls=tool_calls)

    # This should still return the valid tool call but log a warning about the invalid one
    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)

    assert structured_outputs
    assert len(structured_outputs) == 1  # Only the valid tool call
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert text_response == "Test response"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_all_invalid_tools(
    mock_completion, mock_litellm_multi_tool_response, test_schemas, test_messages
):
    """Test validating response where all tool calls are invalid"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # All invalid tool calls
    tool_calls = [
        {
            "name": "TestActionArgs",
            "args": {"invalid_field": "test"}  # Missing required 'command'
        },
        {
            "name": "AnotherActionArgs",
            "args": {"recursive": True}  # Missing required 'path'
        }
    ]
    
    mock_response = mock_litellm_multi_tool_response(tool_calls=tool_calls)

    # Should raise CompletionRetryError as all tools are invalid
    with pytest.raises(CompletionRetryError):
        await model._validate_completion(completion_response=mock_response)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_invalid_wrapper_format(
    mock_completion, test_schemas, test_messages
):
    """Test validating an invalid wrapper format (missing tool_calls)"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # Create a direct invalid wrapper response
    function = Function(name="MultiToolCall", arguments=json.dumps({"invalid_key": "value"}))
    message_tool_calls = [
        ChatCompletionMessageToolCall(function=function, id="invalid-wrapper", type="function")
    ]
    message = Message(content="Test response", tool_calls=message_tool_calls, role="assistant")
    
    mock_response = ModelResponse(
        id="test_id",
        created=1234567890,
        model="test",
        choices=[{"message": message, "finish_reason": "stop", "index": 0}],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )

    # Should raise CompletionRetryError for invalid wrapper format
    with pytest.raises(CompletionRetryError) as excinfo:
        await model._validate_completion(completion_response=mock_response)
    
    assert "MultiToolCall must contain a 'tool_calls' array" in str(excinfo.value)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_end_to_end_completion(mock_completion, mock_litellm_multi_tool_response, test_schemas, test_messages):
    """Test complete multi-tool call completion flow with wrapper"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Execute the commands")

    # Create tool calls for the wrapper
    tool_calls = [
        {
            "name": "TestActionArgs",
            "args": {"command": "test", "args": ["--flag"], "thoughts": "Testing command execution"}
        },
        {
            "name": "AnotherActionArgs",
            "args": {"path": "/tmp/test", "recursive": True, "thoughts": "Testing path processing"}
        }
    ]
    
    mock_response = mock_litellm_multi_tool_response(tool_calls=tool_calls)
    mock_completion.return_value = mock_response

    result = await model.create_completion(messages=test_messages)

    assert result.structured_outputs
    assert len(result.structured_outputs) == 2
    
    # Verify both tool calls are present in the result by checking instance types
    has_test_action = False
    has_another_action = False
    
    for output in result.structured_outputs:
        if isinstance(output, TestActionArgs):
            has_test_action = True
            assert output.command == "test"
            assert output.args == ["--flag"]
        elif isinstance(output, AnotherActionArgs):
            has_another_action = True
            assert output.path == "/tmp/test"
            assert output.recursive is True
    
    assert has_test_action, "TestActionArgs not found in the outputs"
    assert has_another_action, "AnotherActionArgs not found in the outputs"

    # Verify correct parameters were passed
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert "tools" in call_kwargs
    assert len(call_kwargs["tools"]) == 3  # All original tools plus wrapper
    assert call_kwargs["tools"][2]["function"]["name"] == "MultiToolCall"
    assert len(call_kwargs["messages"]) == len(test_messages) + 1  # +1 for system prompt 


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_direct_tool_calls(mock_completion, test_schemas, test_messages):
    """Test handling direct tool calls without using the MultiToolCall wrapper"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # Create direct tool call without wrapper
    function = Function(name="TestActionArgs", arguments=json.dumps({
        "command": "test",
        "args": ["--flag"]
    }))
    
    message_tool_calls = [
        ChatCompletionMessageToolCall(function=function, id="direct-call", type="function")
    ]
    
    message = Message(content="Test response", tool_calls=message_tool_calls, role="assistant")
    
    mock_response = ModelResponse(
        id="test_id",
        created=1234567890,
        model="test",
        choices=[{"message": message, "finish_reason": "stop", "index": 0}],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    
    mock_completion.return_value = mock_response

    # Process the direct tool call
    result = await model.create_completion(messages=test_messages)

    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert isinstance(result.structured_outputs[0], TestActionArgs)
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"] 


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_multiple_direct_tool_calls(mock_completion, test_schemas, test_messages):
    """Test handling multiple direct tool calls without using the MultiToolCall wrapper"""
    model = MultiToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schemas, system_prompt="Test prompt")

    # Create multiple direct tool calls
    message_tool_calls = [
        ChatCompletionMessageToolCall(
            function=Function(name="TestActionArgs", arguments=json.dumps({
                "command": "test",
                "args": ["--flag"]
            })),
            id="direct-call-1", 
            type="function"
        ),
        ChatCompletionMessageToolCall(
            function=Function(name="AnotherActionArgs", arguments=json.dumps({
                "path": "/tmp/test",
                "recursive": True
            })),
            id="direct-call-2", 
            type="function"
        )
    ]
    
    message = Message(content="Test response", tool_calls=message_tool_calls, role="assistant")
    
    mock_response = ModelResponse(
        id="test_id",
        created=1234567890,
        model="test",
        choices=[{"message": message, "finish_reason": "stop", "index": 0}],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    
    mock_completion.return_value = mock_response

    # Process the direct tool calls
    result = await model.create_completion(messages=test_messages)

    assert result.structured_outputs
    assert len(result.structured_outputs) == 2
    
    # Verify both tool calls are present in the result by checking instance types
    has_test_action = False
    has_another_action = False
    
    for output in result.structured_outputs:
        if isinstance(output, TestActionArgs):
            has_test_action = True
            assert output.command == "test"
            assert output.args == ["--flag"]
        elif isinstance(output, AnotherActionArgs):
            has_another_action = True
            assert output.path == "/tmp/test"
            assert output.recursive is True
    
    assert has_test_action, "TestActionArgs not found in the outputs"
    assert has_another_action, "AnotherActionArgs not found in the outputs" 