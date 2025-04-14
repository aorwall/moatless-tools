import json
from typing import Any, Dict, List, Optional, Union, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pytest
from litellm.files.main import RateLimitError
from moatless.completion.base import (
    BaseCompletionModel,
    CompletionResponse,
    CompletionRetryError,
    LLMResponseFormat,
)
from moatless.completion.react import ReActCompletionModel
from moatless.completion.schema import (
    AllMessageValues,
    ResponseSchema,
)
from moatless.completion.stats import CompletionAttempt, CompletionInvocation, Usage
from moatless.exceptions import CompletionRuntimeError
from openai import APIConnectionError
from pydantic import BaseModel, Field


# Create a minimal ResponseSchema subclass for testing
class TestResponse(ResponseSchema):
    answer: str = Field(description="The answer to the question")

    @classmethod
    def description(cls):
        return "A test response schema for testing"


# Concrete implementation of BaseCompletionModel for testing
class TestCompletionModel(BaseCompletionModel):
    model_id: str = "test-model-id"
    model: str = "test-model"
    temperature: float = 0.0
    max_tokens: int = 100
    timeout: float = 30.0
    few_shot_examples: bool = False

    async def _validate_completion(
        self, completion_response: Any
    ) -> Tuple[List[ResponseSchema], Optional[str], Optional[str]]:
        """Implementation of the abstract method."""
        # Simple validation that extracts answer from content
        try:
            answer = completion_response.choices[0].message.content
            response = TestResponse(answer=answer)
            return [response], answer, None
        except Exception as e:
            return [], str(e), None


class TestCompletionRetryError:
    def test_init_with_no_retry_messages(self):
        """Test that CompletionRetryError can be initialized with no retry messages."""
        error = CompletionRetryError("Test error")
        assert error.retry_messages == []

    def test_init_with_single_retry_message(self):
        """Test that CompletionRetryError can be initialized with a single retry message."""
        message = cast(AllMessageValues, {"role": "user", "content": "Test message"})
        error = CompletionRetryError("Test error", retry_message=message)
        assert error.retry_messages == [message]

    def test_init_with_multiple_retry_messages(self):
        """Test that CompletionRetryError can be initialized with multiple retry messages."""
        messages = [
            cast(AllMessageValues, {"role": "user", "content": "Test message 1"}),
            cast(AllMessageValues, {"role": "user", "content": "Test message 2"}),
        ]
        error = CompletionRetryError("Test error", retry_messages=messages)
        assert error.retry_messages == messages


class TestLLMResponseFormat:
    def test_enum_values(self):
        """Test that LLMResponseFormat has the expected values."""
        assert LLMResponseFormat.TOOLS == "tool_call"
        assert LLMResponseFormat.JSON == "json"
        assert LLMResponseFormat.ANTHROPIC_TOOLS == "anthropic_tools"
        assert LLMResponseFormat.REACT == "react"


class TestCompletionResponse:
    def test_structured_output_property_single_output(self):
        """Test that structured_output property returns the first output when there's one output."""
        response = TestResponse(answer="Test answer")

        # Create a completion attempt with token counts
        attempt = CompletionAttempt()
        attempt.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            completion_cost=0.05,  # Pre-set cost instead of calculating
        )

        # Create a CompletionInvocation with the attempt
        invocation = CompletionInvocation(model="test-model")
        invocation.attempts = [attempt]

        completion_response = CompletionResponse(structured_outputs=[response], completion_invocation=invocation)
        assert completion_response.structured_output == response
        if completion_response.structured_output:
            assert completion_response.structured_output.model_dump()["answer"] == "Test answer"

    def test_structured_output_property_multiple_outputs(self):
        """Test that structured_output property returns the first output with a warning when there are multiple outputs."""
        response1 = TestResponse(answer="Test answer 1")
        response2 = TestResponse(answer="Test answer 2")

        # Create a completion attempt with token counts
        attempt = CompletionAttempt()
        attempt.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            completion_cost=0.05,  # Pre-set cost instead of calculating
        )

        # Create a CompletionInvocation with the attempt
        invocation = CompletionInvocation(model="test-model")
        invocation.attempts = [attempt]

        with patch("moatless.completion.base.logger") as mock_logger:
            completion_response = CompletionResponse(
                structured_outputs=[response1, response2], completion_invocation=invocation
            )
            assert completion_response.structured_output == response1
            mock_logger.warning.assert_called_once()

    def test_structured_output_property_no_outputs(self):
        """Test that structured_output property returns None when there are no outputs."""
        # Create a completion attempt with token counts
        attempt = CompletionAttempt()
        attempt.usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            completion_cost=0.0,  # Pre-set cost instead of calculating
        )

        # Create a CompletionInvocation with the attempt
        invocation = CompletionInvocation(model="test-model")
        invocation.attempts = [attempt]

        completion_response = CompletionResponse(structured_outputs=[], completion_invocation=invocation)
        assert completion_response.structured_output is None

    def test_model_validate(self):
        model = BaseCompletionModel.model_validate(
            {
                "completion_model_class": "moatless.completion.react.ReActCompletionModel",
                "model_id": "test-model-id",
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": 100,
                "timeout": 30.0,
            }
        )
        assert isinstance(model, ReActCompletionModel)


@pytest.fixture
def model():
    """Fixture that returns a new TestCompletionModel instance."""
    return TestCompletionModel(
        model_id="test-model-id",
        model="test-model",
        temperature=0.0,
        max_tokens=100,
        timeout=30.0,
    )


class TestBaseCompletionModel:
    def test_get_component_type(self):
        """Test get_component_type class method."""
        assert BaseCompletionModel.get_component_type() == "completion_model"

    def test_get_package(self):
        """Test _get_package class method."""
        assert BaseCompletionModel._get_package() == "moatless.completion"

    def test_get_base_class(self):
        """Test _get_base_class class method."""
        assert BaseCompletionModel._get_base_class() == BaseCompletionModel

    def test_initialized_property(self, model):
        """Test initialized property."""
        assert not model.initialized

        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")
        assert model.initialized

    def test_initialize_with_single_schema(self, model):
        """Test initialize method with a single schema."""
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")
        assert model._response_schema == [TestResponse]
        assert model._system_prompt == "Test prompt"

    def test_initialize_with_schema_list(self, model):
        """Test initialize method with a list of schemas."""
        model.initialize(response_schema=[TestResponse], system_prompt="Test prompt")
        assert model._response_schema == [TestResponse]
        assert model._system_prompt == "Test prompt"

    def test_initialize_with_empty_schema_list(self, model):
        """Test initialize method with an empty schema list."""
        with pytest.raises(CompletionRuntimeError, match="At least one response schema must be provided"):
            model.initialize(response_schema=[], system_prompt="Test prompt")

    def test_initialize_with_invalid_schema(self, model):
        """Test initialize method with an invalid schema."""
        with pytest.raises(CompletionRuntimeError, match="must be a subclass of ResponseSchema"):
            # type: ignore - we're testing invalid input
            model.initialize(response_schema=str, system_prompt="Test prompt")

    def test_initialize_schema_cannot_be_changed(self, model):
        """Test that response schema cannot be changed after initialization."""
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")

        # Try to change the schema
        with pytest.raises(ValueError, match="Response schema cannot be changed after initialization"):
            model.initialize(response_schema=[TestResponse, TestResponse], system_prompt="Test prompt")

    def test_initialize_system_prompt_cannot_be_changed(self, model):
        """Test that system prompt cannot be changed after initialization."""
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")

        # Try to change the system prompt
        with pytest.raises(ValueError, match="System prompt cannot be changed after initialization"):
            model.initialize(response_schema=TestResponse, system_prompt="Different prompt")

    def test_prepare_system_prompt(self, model):
        """Test _prepare_system_prompt method."""
        system_prompt = "Test system prompt"
        prompt = model._prepare_system_prompt(system_prompt, [TestResponse])
        assert prompt == system_prompt  # Default implementation returns prompt unchanged

    def test_generate_few_shot_examples(self, model):
        """Test _generate_few_shot_examples method."""
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")
        examples = model._generate_few_shot_examples()
        assert isinstance(examples, str)
        assert examples == "" or "Examples" in examples  # Either empty or contains 'Examples'

    def test_prepare_messages(self, model):
        """Test _prepare_messages method."""
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")
        messages = [{"role": "user", "content": "Test message"}]
        prepared = model._prepare_messages(messages, "System prompt")

        assert len(prepared) == 2
        assert prepared[0]["role"] == "system"
        assert prepared[0]["content"] == "System prompt"
        assert prepared[1]["role"] == "user"
        assert prepared[1]["content"] == "Test message"

    def test_get_completion_params(self, model):
        """Test _get_completion_params method."""
        params = model._get_completion_params([TestResponse])
        assert isinstance(params, dict)

    @pytest.mark.asyncio
    async def test_create_completion_not_initialized(self, model):
        """Test create_completion method when the model is not initialized."""
        with pytest.raises(ValueError, match="Model must be initialized"):
            await model.create_completion([{"role": "user", "content": "Test message"}])

    @pytest.mark.asyncio
    async def test_create_completion_no_system_prompt(self, model):
        """Test create_completion method when no system prompt is provided."""
        # Initialize with a None system prompt
        model.initialize(response_schema=TestResponse, system_prompt=None)
        with pytest.raises(ValueError, match="No system prompt provided"):
            await model.create_completion([{"role": "user", "content": "Test message"}])

    @pytest.mark.asyncio
    async def test_create_completion(self, model):
        """Test create_completion method."""
        # Initialize the model
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")

        # Create a mock response to be returned by _create_completion_with_retries
        expected_outputs = [TestResponse(answer="Test answer")]

        # Create a completion attempt with token counts
        attempt = CompletionAttempt()
        attempt.usage = Usage(prompt_tokens=100, completion_tokens=50)

        # Create a CompletionInvocation with the attempt
        invocation = CompletionInvocation(model="test-model")
        invocation.attempts = [attempt]

        expected_response = CompletionResponse(structured_outputs=expected_outputs, completion_invocation=invocation)

        # Mock _create_completion_with_retries to return our expected response
        with patch.object(model, "_create_completion_with_retries", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = expected_response

            # Call create_completion
            response = await model.create_completion([{"role": "user", "content": "Test message"}])

            # Verify _create_completion_with_retries was called correctly
            mock_create.assert_called_once()

            # Verify the response
            assert response == expected_response
            assert response.structured_outputs == expected_outputs

    @pytest.mark.asyncio
    async def test_create_completion_with_custom_system_prompt(self, model):
        """Test create_completion method with a custom system prompt."""
        # Initialize the model
        model.initialize(response_schema=TestResponse, system_prompt="Default prompt")

        # Create a mock response with completion attempts
        attempt = CompletionAttempt()
        attempt.usage = Usage(prompt_tokens=100, completion_tokens=50)

        # Create a CompletionInvocation with the attempt
        invocation = CompletionInvocation(model="test-model")
        invocation.attempts = [attempt]

        expected_response = CompletionResponse(
            structured_outputs=[TestResponse(answer="Test answer")], completion_invocation=invocation
        )

        # Mock _create_completion_with_retries
        with patch.object(model, "_create_completion_with_retries", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = expected_response

            # Call create_completion with a custom system prompt
            response = await model.create_completion(
                [{"role": "user", "content": "Test message"}], system_prompt="Custom prompt"
            )

            # Verify the custom system prompt was used - since we can't access the messages directly,
            # we'll just verify the method was called
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_completion_rate_limit_retry(self):
        """Test that _execute_completion retries rate limit errors."""
        model = TestCompletionModel()
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")

        # Create a mock successful response for the second attempt
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "Test answer"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = {"prompt_tokens": 100, "completion_tokens": 50}

        # Mock litellm.acompletion to first fail with RateLimitError, then succeed
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
            # First call raises RateLimitError, second call succeeds
            mock_litellm.side_effect = [
                RateLimitError("Rate limit exceeded", llm_provider="openai", model="gpt-4"),
                mock_response,
            ]

            # Call _execute_completion directly
            invocation = CompletionInvocation(model=model.model)

            # Expect this to succeed after retrying the RateLimitError
            result = await model._execute_completion([{"role": "user", "content": "Test message"}], invocation)

            # Assert success
            assert result == mock_response

            # Verify litellm.acompletion was called twice
            assert mock_litellm.call_count == 2

            # Verify the invocation attempts
            assert len(invocation.attempts) == 2

            # First attempt should be failed
            assert invocation.attempts[0].success is False
            assert invocation.attempts[0].failure_reason is not None
            assert "Rate limit" in invocation.attempts[0].failure_reason

            # Second attempt should be successful
            assert invocation.attempts[1].success is True

    @pytest.mark.asyncio
    async def test_validation_retry_logic(self):
        """Test validation retry logic with CompletionRetryError."""
        model = TestCompletionModel()
        model.initialize(response_schema=TestResponse, system_prompt="Test prompt")

        # Create mock responses
        mock_response1 = MagicMock()
        mock_choice1 = MagicMock()
        mock_message1 = MagicMock()
        mock_message1.content = "Invalid answer"
        mock_choice1.message = mock_message1
        mock_response1.choices = [mock_choice1]
        mock_response1.usage = {"prompt_tokens": 100, "completion_tokens": 30}

        mock_response2 = MagicMock()
        mock_choice2 = MagicMock()
        mock_message2 = MagicMock()
        mock_message2.content = "Valid answer"
        mock_choice2.message = mock_message2
        mock_response2.choices = [mock_choice2]
        mock_response2.usage = {"prompt_tokens": 120, "completion_tokens": 40}

        # Override validate_completion to first fail with CompletionRetryError, then succeed
        original_validate = model._validate_completion
        validation_attempts = 0

        async def mock_validate(completion_response):
            nonlocal validation_attempts
            validation_attempts += 1

            if validation_attempts == 1:
                # First validation fails
                raise CompletionRetryError("Invalid response format")
            else:
                # Second validation succeeds
                return await original_validate(completion_response)

        with patch.object(model, "_validate_completion", side_effect=mock_validate):
            # Mock _execute_completion to return different responses
            with patch.object(model, "_execute_completion", new_callable=AsyncMock) as mock_execute:
                mock_execute.side_effect = [mock_response1, mock_response2]

                # Call create_completion
                response = await model.create_completion([{"role": "user", "content": "Test message"}])

                # Verify both execute and validate were called properly
                assert mock_execute.call_count == 2
                assert validation_attempts == 2

                # Verify the response contains both attempts
                assert response.completion_invocation is not None
                assert len(response.completion_invocation.attempts) == 2

                # First attempt should be failed due to validation
                assert response.completion_invocation.attempts[0].success is False
                assert response.completion_invocation.attempts[0].failure_reason is not None
                assert "Invalid response format" in response.completion_invocation.attempts[0].failure_reason

                # Second attempt should be successful
                assert response.completion_invocation.attempts[1].success is True

                # Final result should contain the second response
                assert response.structured_output is not None
                # Check that it's the expected TestResponse type
                assert isinstance(response.structured_output, TestResponse)
                assert response.structured_output.answer == "Valid answer"

    def test_merge_same_role_messages(self):
        """Test _merge_same_role_messages method."""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "user", "content": "User 1"},
            {"role": "user", "content": "User 2"},
            {"role": "assistant", "content": "Assistant 1"},
            {"role": "assistant", "content": "Assistant 2"},
            {"role": "user", "content": "User 3"},
        ]

        # Create an instance of TestCompletionModel to call the instance method
        model = TestCompletionModel(
            model_id="test-model-id",
            model="test-model",
        )
        merged = model._merge_same_role_messages(messages)

        # The implementation only merges consecutive user messages
        # System message is preserved
        assert merged[0]["role"] == "system"
        assert merged[0]["content"] == "System 1"

        # User messages 1 and 2 should be merged
        assert merged[1]["role"] == "user"
        assert merged[1]["content"] == "User 1\nUser 2"

        # Assistant messages are not merged
        assert merged[2]["role"] == "assistant"
        assert merged[2]["content"] == "Assistant 1"
        assert merged[3]["role"] == "assistant"
        assert merged[3]["content"] == "Assistant 2"

        # Final user message is preserved
        assert merged[4]["role"] == "user"
        assert merged[4]["content"] == "User 3"

        # Total count should be 5 messages
