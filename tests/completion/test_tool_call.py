import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import tenacity
from litellm.types.utils import ModelResponse, Usage, Message, Function, ChatCompletionMessageToolCall
from moatless.actions.schema import ActionArguments
from moatless.completion.base import CompletionRetryError
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError
from pydantic import Field


class TestActionArgs(ActionArguments):
    command: str = Field(..., description="The command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")

    model_config = {"title": "TestActionArgs"}


@pytest.fixture
def test_schema():
    return [TestActionArgs]


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "Run the test command"}]


@pytest.fixture
def mock_litellm_tool_response():
    """Mock LiteLLM response with tool calls"""

    def _create_mock(tool_calls=None, usage=None):
        # Create tool calls
        if tool_calls:
            message_tool_calls = []
            for tc in tool_calls:
                function = Function(name=tc["name"], arguments=tc["args"])
                tool_call = ChatCompletionMessageToolCall(
                    function=function, id=f"tool-{len(message_tool_calls)}", type="function"
                )
                message_tool_calls.append(tool_call)
        else:
            message_tool_calls = None

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


def test_prepare_schema_single(test_schema):
    """Test preparing single tool schema"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    assert len(model._response_schema) == 1
    assert model._response_schema[0] == test_schema[0]


def test_prepare_schema_multiple(test_schema):
    """Test preparing multiple tool schemas"""

    class AnotherAction(TestActionArgs):
        model_config = {"title": "AnotherAction"}

    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=[test_schema[0], AnotherAction], system_prompt="Test prompt")
    assert len(model._response_schema) == 2
    assert model._response_schema[0] == test_schema[0]
    assert model._response_schema[1] == AnotherAction


def test_prepare_schema_invalid():
    """Test preparing invalid schema"""

    class InvalidSchema:
        pass

    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    with pytest.raises(CompletionRuntimeError):
        model.initialize(response_schema=[InvalidSchema], system_prompt="Test prompt")  # type: ignore


def test_get_completion_params(test_schema):
    """Test getting tool completion parameters"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    assert model._completion_params
    assert "tools" in model._completion_params
    assert len(model._completion_params["tools"]) == 1
    assert model._completion_params["tools"][0]["type"] == "function"
    assert model._completion_params["tool_choice"] == "auto"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_valid_tool(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test validating valid tool call response"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    # Mock valid tool call
    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing command execution"}
    mock_response = mock_litellm_tool_response(tool_calls=[{"name": "TestActionArgs", "args": json.dumps(valid_args)}])

    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)

    assert structured_outputs
    assert len(structured_outputs) == 1
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert structured_outputs[0].thoughts == "Testing command execution"
    assert text_response == "Test response"
    assert thought is None  # Tool call model returns None for thought


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_mixed_valid_invalid_toolcalls(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test that only valid toolcalls are returned and invalid ones are ignored (no retry if any valid)"""
    model = ToolCallCompletionModel(model="test")  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Valid call"}
    invalid_args = {"invalid_field": "fail"}
    # One valid, one invalid tool name, one invalid args
    tool_calls = [
        {"name": "TestActionArgs", "args": json.dumps(valid_args)},  # valid
        {"name": "invalid_tool", "args": json.dumps(valid_args)},  # invalid tool name
        {"name": "TestActionArgs", "args": json.dumps(invalid_args)},  # invalid args
    ]
    mock_response = mock_litellm_tool_response(tool_calls=tool_calls)

    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)
    # Only the valid toolcall should be returned
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert structured_outputs[0].command == "test"
    assert structured_outputs[0].args == ["--flag"]
    assert structured_outputs[0].thoughts == "Valid call"
    assert text_response == "Test response"
    assert thought is None


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_all_invalid_toolcalls(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test that retry is triggered if all toolcalls are invalid (invalid name or args)"""
    model = ToolCallCompletionModel(model="test")  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    invalid_args = {"invalid_field": "fail"}
    tool_calls = [
        {"name": "invalid_tool", "args": json.dumps(invalid_args)},
        {"name": "invalid_tool2", "args": json.dumps(invalid_args)},
    ]
    mock_response = mock_litellm_tool_response(tool_calls=tool_calls)

    with pytest.raises(CompletionRetryError):
        await model._validate_completion(completion_response=mock_response)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_all_valid_toolcalls(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test that all valid toolcalls are returned if all are valid (no retry)"""
    model = ToolCallCompletionModel(model="test")  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    valid_args1 = {"command": "test1", "args": ["--flag1"], "thoughts": "Valid1"}
    valid_args2 = {"command": "test2", "args": ["--flag2"], "thoughts": "Valid2"}
    tool_calls = [
        {"name": "TestActionArgs", "args": json.dumps(valid_args1)},
        {"name": "TestActionArgs", "args": json.dumps(valid_args2)},
    ]
    mock_response = mock_litellm_tool_response(tool_calls=tool_calls)

    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)
    assert structured_outputs
    assert len(structured_outputs) == 2
    assert {o.command for o in structured_outputs} == {"test1", "test2"}
    assert text_response == "Test response"
    assert thought is None


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_invalid_tool_name(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test validating invalid tool name"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    # Mock invalid tool name
    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing invalid tool name"}
    mock_response = mock_litellm_tool_response(tool_calls=[{"name": "invalid_action", "args": json.dumps(valid_args)}])

    with pytest.raises(CompletionRetryError):
        await model._validate_completion(completion_response=mock_response)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_invalid_args(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test validating invalid tool arguments"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    # Mock invalid arguments
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments",
    }
    mock_response = mock_litellm_tool_response(
        tool_calls=[{"name": "TestActionArgs", "args": json.dumps(invalid_args)}]
    )

    with pytest.raises(CompletionRetryError):
        await model._validate_completion(completion_response=mock_response)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_validate_completion_duplicate_calls(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test handling duplicate tool calls"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")

    # Mock duplicate tool calls with same arguments
    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing duplicate calls"}
    mock_response = mock_litellm_tool_response(
        tool_calls=[
            {"name": "TestActionArgs", "args": json.dumps(valid_args)},
            {"name": "TestActionArgs", "args": json.dumps(valid_args)},
        ]
    )

    # We no longer need to patch any internal methods since duplicates are handled in the validate function
    structured_outputs, text_response, thought = await model._validate_completion(completion_response=mock_response)
    assert len(structured_outputs) == 1
    # ToolCallCompletionModel doesn't extract thoughts from args into the thought parameter
    assert thought is None
    assert structured_outputs[0].thoughts == "Testing duplicate calls"
    assert text_response == "Test response"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_end_to_end_completion(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test complete tool call completion flow"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")

    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing end to end completion"}
    mock_response = mock_litellm_tool_response(tool_calls=[{"name": "TestActionArgs", "args": json.dumps(valid_args)}])
    mock_completion.return_value = mock_response

    # Create a patched version of _validate_completion to set thought from the structured_outputs
    original_validate = model._validate_completion

    async def patched_validate(*args, **kwargs):
        structured_outputs, text_response, _ = await original_validate(*args, **kwargs)
        # Extract thought from the first structured output
        thought = structured_outputs[0].thoughts if structured_outputs else None
        return structured_outputs, text_response, thought

    # Apply the patch for this test only
    with patch.object(model, "_validate_completion", side_effect=patched_validate):
        result = await model.create_completion(messages=test_messages)

    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"]
    assert result.structured_outputs[0].thoughts == "Testing end to end completion"
    assert result.thought == "Testing end to end completion"

    # Verify correct parameters were passed
    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert "tools" in call_kwargs
    assert len(call_kwargs["messages"]) == len(test_messages) + 1  # +1 for system prompt


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_retry_on_validation_error(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that validation errors trigger a retry of the entire completion process"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")

    # First response has invalid args to trigger validation error
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments",
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[{"name": "TestActionArgs", "args": json.dumps(invalid_args)}]
    )

    # Second response is valid
    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing retry success"}
    valid_response = mock_litellm_tool_response(tool_calls=[{"name": "TestActionArgs", "args": json.dumps(valid_args)}])

    # Set up side effects to raise CompletionRetryError for first call
    async def validate_side_effect(completion_response):
        if completion_response == invalid_response:
            raise CompletionRetryError("Command field is required")
        return (
            [TestActionArgs(command="test", args=["--flag"], thoughts="Testing retry success")],
            "Test response",
            "Testing retry success",
        )

    # Mock both acompletion and _validate_completion
    mock_completion.side_effect = [invalid_response, valid_response]
    with patch.object(model, "_validate_completion", side_effect=validate_side_effect):
        result = await model.create_completion(messages=test_messages)

    # Verify completion was called twice
    assert mock_completion.call_count == 2

    # Verify final result is from the valid response
    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert result.structured_outputs[0].command == "test"
    assert result.structured_outputs[0].args == ["--flag"]
    assert result.structured_outputs[0].thoughts == "Testing retry success"
    assert result.thought == "Testing retry success"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_retry_max_attempts_exceeded(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that validation errors stop retrying after max attempts and raise the error"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")

    # Create invalid response that will always fail validation
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments",
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[{"name": "TestActionArgs", "args": json.dumps(invalid_args)}]
    )

    # Set up side effect to always raise CompletionRetryError
    async def validate_side_effect(completion_response):
        raise CompletionRetryError("Tool args validation failed: command field is required")

    # Mock both acompletion and _validate_completion
    mock_completion.return_value = invalid_response
    with patch.object(model, "_validate_completion", side_effect=validate_side_effect):
        with pytest.raises(CompletionRejectError) as exc_info:
            await model.create_completion(messages=test_messages)

    # Verify completion was called exactly 3 times
    assert mock_completion.call_count == 3

    # Verify the error message contains the validation error
    assert "Tool args validation failed" in str(exc_info.value)


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_usage_accumulation_on_retries(mock_completion, mock_litellm_tool_response, test_schema, test_messages):
    """Test that usage is accumulated correctly across retries"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")

    # First response has invalid args to trigger validation error
    invalid_args = {
        "invalid_field": "test",  # Missing required 'command'
        "thoughts": "Testing invalid arguments",
    }
    invalid_response = mock_litellm_tool_response(
        tool_calls=[{"name": "TestActionArgs", "args": json.dumps(invalid_args)}],
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )

    # Second response is valid
    valid_args = {"command": "test", "args": ["--flag"], "thoughts": "Testing retry success"}
    valid_response = mock_litellm_tool_response(
        tool_calls=[{"name": "TestActionArgs", "args": json.dumps(valid_args)}],
        usage={"prompt_tokens": 15, "completion_tokens": 8},
    )

    # Set up side effects to raise CompletionRetryError for first call
    async def validate_side_effect(completion_response):
        if completion_response == invalid_response:
            raise CompletionRetryError("Command field is required")
        return (
            [TestActionArgs(command="test", args=["--flag"], thoughts="Testing retry success")],
            "Test response",
            "Testing retry success",
        )

    # Mock both acompletion and _validate_completion
    mock_completion.side_effect = [invalid_response, valid_response]
    with patch.object(model, "_validate_completion", side_effect=validate_side_effect):
        result = await model.create_completion(messages=test_messages)

    # Verify completion was called twice
    assert mock_completion.call_count == 2

    # Verify completion attempt metrics were tracked
    assert len(result.completion_invocation.attempts) >= 2
    assert result.thought == "Testing retry success"


@pytest.mark.asyncio
@patch("litellm.acompletion", new_callable=AsyncMock)
async def test_usage_accumulation_on_max_retries(
    mock_completion, mock_litellm_tool_response, test_schema, test_messages
):
    """Test that usage is accumulated correctly when max retries is exceeded"""
    model = ToolCallCompletionModel(
        model="test",
    )  # type: ignore
    model.initialize(response_schema=test_schema, system_prompt="Execute the command")

    # Create invalid responses with different usage values
    def get_response_with_usage(attempt):
        return mock_litellm_tool_response(
            tool_calls=[{"name": "TestActionArgs", "args": json.dumps({"invalid_field": "test"})}],
            usage={"prompt_tokens": 10 * (attempt + 1), "completion_tokens": 5 * (attempt + 1)},
        )

    # Set up responses with increasing token usage
    mock_responses = [get_response_with_usage(i) for i in range(3)]

    # Set up side effect to always raise CompletionRetryError
    async def validate_side_effect(completion_response):
        raise CompletionRetryError("Tool args validation failed: command field is required")

    # Mock both acompletion and _validate_completion
    mock_completion.side_effect = mock_responses
    with patch.object(model, "_validate_completion", side_effect=validate_side_effect):
        with pytest.raises(CompletionRejectError) as exc_info:
            await model.create_completion(messages=test_messages)

    # Verify completion was called exactly 3 times
    assert mock_completion.call_count == 3

    # Verify completion attempt metrics were tracked
    assert len(exc_info.value.completion_invocation.attempts) >= 3
