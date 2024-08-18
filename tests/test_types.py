import anthropic.types
from moatless.schema import Completion, Usage

class TestCompletion:
    def test_from_llm_completion_with_dict_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {
            "choices": [{"message": {"content": "Test output"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(input_messages, completion_response, model)

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 5

    def test_from_llm_completion_with_anthropic_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = anthropic.types.Message(
            id="msg_123",
            model="claude-3.5-sonnet",
            type="message",
            role="assistant",
            content=[
                anthropic.types.TextBlock(text="Test output", type="text"),
                anthropic.types.ToolUseBlock(id="tool_1", input={"query": "test"}, name="search", type="tool_use")
            ],
            usage=anthropic.types.Usage(input_tokens=10, output_tokens=20)
        )
        model = "claude-3.5-sonnet"

        completion = Completion.from_llm_completion(input_messages, completion_response, model)

        assert completion.input == input_messages
        assert completion.response == completion_response.model_dump()
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 20

    def test_from_llm_completion_with_missing_usage(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {
            "choices": [{"message": {"content": "Test output"}}]
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(input_messages, completion_response, model)

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert completion.usage is None

    def test_from_llm_completion_with_unexpected_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = "Unexpected string response"
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(input_messages, completion_response, model)

        assert completion is None

    def test_from_llm_completion_with_multiple_messages(self):
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        completion_response = {
            "choices": [{"message": {"content": "I'm doing well, thank you for asking!"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(input_messages, completion_response, model)

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 20
        assert completion.usage.completion_tokens == 10