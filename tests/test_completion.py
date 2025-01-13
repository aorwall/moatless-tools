import json
from typing import Optional
from unittest.mock import patch, MagicMock

import anthropic.types
import pytest
from litellm.types.utils import ModelResponse

from moatless.actions.create_file import CreateFileArgs
from moatless.actions.find_function import FindFunctionArgs
from moatless.actions.model import ActionArguments
from moatless.actions.string_replace import StringReplaceArgs
from moatless.completion.completion import CompletionModel, LLMResponseFormat, CompletionResponse
from moatless.completion.model import Usage, Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError


class TestCompletion:
    def test_from_llm_completion_with_dict_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {
            "choices": [{"message": {"content": "Test output"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

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
                anthropic.types.ToolUseBlock(
                    id="tool_1", input={"query": "test"}, name="search", type="tool_use"
                ),
            ],
            usage=anthropic.types.Usage(input_tokens=10, output_tokens=20),
        )
        model = "claude-3.5-sonnet"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response.model_dump()
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 20

    def test_from_llm_completion_with_missing_usage(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = {"choices": [{"message": {"content": "Test output"}}]}
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert completion.usage is None

    def test_from_llm_completion_with_unexpected_response(self):
        input_messages = [{"role": "user", "content": "Test input"}]
        completion_response = "Unexpected string response"
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion is None

    def test_from_llm_completion_with_multiple_messages(self):
        input_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        completion_response = {
            "choices": [
                {"message": {"content": "I'm doing well, thank you for asking!"}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        model = "gpt-3.5-turbo"

        completion = Completion.from_llm_completion(
            input_messages, completion_response, model
        )

        assert completion.input == input_messages
        assert completion.response == completion_response
        assert isinstance(completion.usage, Usage)
        assert completion.usage.prompt_tokens == 20
        assert completion.usage.completion_tokens == 10

    def test_litellm_react_completion(self):
        # Mock action class
        class TestAction(ActionArguments):
            query: str

            class Config:
                title = "TestAction"

        mock_content = """Thought: I need to calculate the sum of 2 and 2.
This is a simple arithmetic operation.
The answer is 4.

Action: TestAction
Action Input: {"query": "2 + 2 = 4"}"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        # Mock _litellm_text_completion
        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            result = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                system_prompt="You are a helpful assistant",
                actions=[TestAction]
            )

            # Verify the parsed output is now a CompletionResponse
            assert isinstance(result, CompletionResponse)
            assert isinstance(result.structured_output, TestAction)
            assert result.structured_output.query == "2 + 2 = 4"
            assert result.structured_output.scratch_pad == (
                "I need to calculate the sum of 2 and 2.\n"
                "This is a simple arithmetic operation.\n"
                "The answer is 4."
            )
            assert result.completion == mock_response

    def test_litellm_react_completion_invalid_format(self):
        class TestAction(StructuredOutput):
            name = "test_action"
            query: str

        mock_response = MagicMock()
        mock_response.choices = [{
            "message": {
                "content": "The answer is 4"  # Invalid format
            }
        }]

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=mock_response):
            with pytest.raises(CompletionRejectError) as exc_info:
                completion._litellm_react_completion(
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    system_prompt="You are a helpful assistant",
                    actions=[TestAction]
                )
            
            assert "Missing Thought, Action or Action Input sections" in str(exc_info.value)

    def test_litellm_react_completion_invalid_action(self):
        class TestAction(StructuredOutput):
            name = "test_action"
            query: str

        mock_response = MagicMock()
        mock_response.choices = [{
            "message": {
                "content": """Thought: Let me calculate this.
Action: unknown_action
Action Input: {"query": "4"}"""
            }
        }]

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=mock_response):
            with pytest.raises(CompletionRejectError) as exc_info:
                completion._litellm_react_completion(
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    system_prompt="You are a helpful assistant",
                    actions=[TestAction]
                )
            
            assert "Unknown action: unknown_action" in str(exc_info.value)

    def test_litellm_react_completion_with_none_value(self):
        class TestAction(StructuredOutput):
            name = "test_action"
            query: str
            optional_field: Optional[str] = None

        mock_content = """Thought: Testing None values
Action: test_action
Action Input: {"query": "test", "optional_field": None}"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            action_request, response = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "Test None values"}],
                system_prompt="You are a helpful assistant",
                actions=[TestAction]
            )

            assert isinstance(action_request, TestAction)
            assert action_request.query == "test"
            assert action_request.optional_field is None

    def test_litellm_react_completion_with_real_example(self):
        mock_content = """Thought: The error indicates that the `StringReplace` action input format was still incorrect. I will ensure the format is correct and provide the exact path, old_str, and new_str.

Let's add the test case again with the correct format.
Action: StringReplace
Action Input: {"path":"tests/queries/test_q.py","old_str":"        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj","new_str":"        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj\n\n        # Test for non-pickleable object in Q object\n        q1 = Q(x__in={}.keys())\n        q2 = Q(y__in={}.keys())\n        self.assertEqual(q1 | q2, Q(x__in=[], y__in=[]))"}"""

        mock_response = ModelResponse()

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            action_request, response = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "How does separability_matrix handle nested CompoundModels?"}],
                system_prompt="You are a helpful assistant",
                actions=[StringReplaceArgs]
            )

            assert isinstance(action_request, FindFunctionArgs)
            assert action_request.file_pattern == 'astropy/modeling/separable.py'
            assert action_request.function_name == '_separable'
            assert action_request.class_name is None
            assert action_request.scratch_pad == (
                "The `separability_matrix` function calls the `_separable` function to compute the separability. "
                "To understand how it handles nested `CompoundModels`, I need to find and examine the implementation "
                "of the `_separable` function."
            )

    def test_litellm_react_completion_with_apply_change(self):
        # Prepare test data
        input_messages = [{"role": "user", "content": "Fix the JSON input"}]
        system_prompt = "You are a helpful assistant."
        
        # Mock the text completion response
        mock_response_text = '''Thought: The JSON input for the ApplyChange action was invalid due to a missing comma. I will correct the JSON input to ensure it is valid.

Action: ApplyChange
Action Input: {
  "path": "django/db/models/fields/__init__.py",
  "old_str": "'invalid': _(\\"'%(value)s' value has an invalid format. It must be in \\"\n                     \\"[DD] [HH:[MM:]]ss[.uuuuuu] format.\\")",
  "new_str": "'invalid': _(\\"'%(value)s' value has an invalid format. It must be in \\"\n                     \\"[DD] [[HH:]MM:]ss[.uuuuuu] format.\\")"
}
'''

        mock_completion_response = ModelResponse(
            id="test_id",
            choices=[{"message": {"content": mock_response_text}}],
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100}
        )

        # Create completion model instance with mocked text completion
        completion_model = CompletionModel(
            model="test-model",
            response_format=LLMResponseFormat.REACT,
            action=[StringReplaceArgs]
        )

        # Mock the _litellm_text_completion method
        with patch.object(completion_model, '_litellm_text_completion', return_value=(mock_response_text, mock_completion_response)):
            # Call the react completion method
            action_args, response = completion_model._litellm_react_completion(
                messages=input_messages,
                system_prompt=system_prompt,
                actions=[StringReplaceArgs]
            )

            # Verify the response structure
            assert isinstance(action_args, StringReplaceArgs)
            assert action_args.path == "django/db/models/fields/__init__.py"
            assert "'invalid': _(" in action_args.old_str
            assert "[DD] [HH:[MM:]]ss[.uuuuuu] format" in action_args.old_str
            assert "'invalid': _(" in action_args.new_str
            assert "[DD] [[HH:]MM:]ss[.uuuuuu] format" in action_args.new_str
            
            # Verify the completion response
            assert response == mock_completion_response
            assert response.usage["prompt_tokens"] == 50
            assert response.usage["completion_tokens"] == 100

    def test_parse_action_input(self):
        # Prepare test data
        input_messages = [{"role": "user", "content": "Fix the JSON input"}]
        system_prompt = "You are a helpful assistant."

        # Mock the text completion response
        mock_response_text = """Thought: The `_repr_latex_` method is not directly defined for `SeqFormula`. However, we found that the `LaTeX` rendering for `SeqFormula` is handled by the `_print_SeqFormula` method in the `LatexPrinter` class within the `sympy/printing/latex.py` file. This method is responsible for generating the LaTeX representation of `SeqFormula` objects.

The issue arises from the use of `\left\[` and `\right\]` which are escaping the square brackets. We need to modify this method to use `[` and `]` directly without the backslashes.

Let's update the `_print_SeqFormula` method to prevent backslash-escaping the square brackets.
Action: ApplyChange
Action Input: {"path": "sympy/printing/latex.py", "old_str": "r\"\\left\[\" + r\", \".join(self._print(el) for el in printset) + r\"\\right\]\"", "new_str": "r\"[\" + r\", \".join(self._print(el) for el in printset) + r\"]\""}"""
        mock_completion_response = ModelResponse(
            id="test_id",
            choices=[{"message": {"content": mock_response_text}}],
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 100}
        )

        # Create completion model instance with mocked text completion
        completion_model = CompletionModel(
            model="test-model",
            response_format=LLMResponseFormat.REACT,
            action=[StringReplaceArgs]
        )

        # Mock the _litellm_text_completion method
        with patch.object(completion_model, '_litellm_text_completion',
                          return_value=(mock_response_text, mock_completion_response)):
            # Call the react completion method
            action_args, response = completion_model._litellm_react_completion(
                messages=input_messages,
                system_prompt=system_prompt,
                actions=[StringReplaceArgs]
            )

            # Verify the response structure
            assert isinstance(action_args, StringReplaceArgs)
            assert action_args.path == "django/db/models/fields/__init__.py"

    def test_parse_structured_input(self):
        content = """{"path": "tests/queries/test_q.py",
"old_str": "        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj",
"new_str": "        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj\n\n        # Test for non-pickleable object in Q object\n        q1 = Q(x__in={}.keys())\n        q2 = Q(y__in={}.keys())\n        self.assertEqual(q1 | q2, Q(x__in=[], y__in=[]))"}"""

        result = StringReplaceArgs.model_validate_json(content)
        assert result.path == "tests/queries/test_q.py"
        assert result.old_str == "        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj"
        assert result.new_str == "        with self.assertRaisesMessage(TypeError, str(obj)):\n            q & obj\n\n        # Test for non-pickleable object in Q object\n        q1 = Q(x__in={}.keys())\n        q2 = Q(y__in={}.keys())\n        self.assertEqual(q1 | q2, Q(x__in=[], y__in=[]))"

    def test_serialization_deserialization(self):
        model = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            response_format=LLMResponseFormat.TOOLS,
        )

        serialized = model.model_dump()
        assert serialized["response_format"] == "tool_call"

        deserialized = CompletionModel.model_validate(serialized)
        assert deserialized.response_format == LLMResponseFormat.TOOLS

        # Check if it's JSON serializable
        json_string = json.dumps(serialized, indent=2)
        print(json_string)
        assert json_string  # This will raise an error if not serializable

    def test_litellm_react_completion_with_string_replace_xml(self):
        mock_content = """Thought: I need to update the error message in the validation function.

Action: ApplyChange
Action Input:
<path>auth/validator.py</path>
<old_str>
    if not user.is_active:
        raise ValueError("Invalid user")
</old_str>
<new_str>
    if not user.is_active:
        raise ValueError(f"User {user.username} is not active")
</new_str>"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            result = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "Update the error message"}],
                system_prompt="You are a helpful assistant",
                actions=[StringReplaceArgs]
            )

            assert isinstance(result, CompletionResponse)
            assert isinstance(result.structured_output, StringReplaceArgs)
            assert result.structured_output.path == "auth/validator.py"
            assert result.structured_output.old_str == "    if not user.is_active:\n        raise ValueError(\"Invalid user\")"
            assert result.structured_output.new_str == "    if not user.is_active:\n        raise ValueError(f\"User {user.username} is not active\")"
            assert result.structured_output.scratch_pad == "I need to update the error message in the validation function."

    def test_litellm_react_completion_with_create_file_xml(self):
        mock_content = """Thought: I need to create a new configuration file.

Action: CreateFile
Action Input:
<path>config/settings.py</path>
<file_text>
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
</file_text>"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 15, "completion_tokens": 25}
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            action_request, response = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "Create a new settings file"}],
                system_prompt="You are a helpful assistant",
                actions=[CreateFileArgs]
            )

            assert isinstance(action_request, CreateFileArgs)
            assert action_request.path == "config/settings.py"
            assert "BASE_DIR = Path(__file__).resolve().parent.parent" in action_request.file_text
            assert "DEBUG = True" in action_request.file_text
            assert "'ENGINE': 'django.db.backends.sqlite3'" in action_request.file_text
            assert action_request.scratch_pad == "I need to create a new configuration file."

    def test_litellm_react_completion_with_invalid_xml(self):
        mock_content = """Thought: I need to update the code.

Action: StringReplace
Action Input:
<path>auth/validator.py</path>
<old_str>
    if not user.is_active:
        raise ValueError("Invalid user")
</old_str>
<!-- missing new_str tag -->
    if not user.is_active:
        raise ValueError("User is inactive")
<!-- missing closing tag -->"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            with pytest.raises(CompletionRejectError) as exc_info:
                completion._litellm_react_completion(
                    messages=[{"role": "user", "content": "Update the error message"}],
                    system_prompt="You are a helpful assistant",
                    actions=[StringReplaceArgs]
                )
            
            assert "Missing new_str tag" in str(exc_info.value)
            assert "Missing closing tag" in str(exc_info.value)

    def test_litellm_react_completion_with_empty_new_str(self):
        mock_content = """Thought: I need to remove this comment block.

Action: ApplyChange
Action Input:
<path>sympy/utilities/iterables.py</path>
<old_str>    Note that the _same_ dictionary object is returned each time.
    This is for speed:  generating each partition goes quickly,
    taking constant time, independent of n.</old_str>
<new_str></new_str>"""

        mock_response = ModelResponse(
            choices=[],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.REACT
        )

        with patch.object(completion, '_litellm_text_completion', return_value=(mock_content, mock_response)):
            action_request, response = completion._litellm_react_completion(
                messages=[{"role": "user", "content": "Remove the comment block"}],
                system_prompt="You are a helpful assistant",
                actions=[StringReplaceArgs]
            )

            assert isinstance(action_request, StringReplaceArgs)
            assert action_request.path == "sympy/utilities/iterables.py"
            assert action_request.old_str == (
                "    Note that the _same_ dictionary object is returned each time.\n"
                "    This is for speed:  generating each partition goes quickly,\n"
                "    taking constant time, independent of n."
            )
            assert action_request.new_str == ""
            assert action_request.scratch_pad == "I need to remove this comment block."

    def test_litellm_tool_completion(self):
        # Mock action class
        class TestAction(ActionArguments):
            query: str

        mock_response = ModelResponse(
            choices=[{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "TestAction",
                            "arguments": '{"query": "test query"}'
                        }
                    }]
                }
            }],
            model="gpt-3.5-turbo",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )

        completion = CompletionModel(
            model="gpt-3.5-turbo",
            temperature=0.7,
            response_format=LLMResponseFormat.TOOLS
        )

        with patch('litellm.completion', return_value=mock_response):
            result = completion._litellm_tool_completion(
                messages=[{"role": "user", "content": "Test query"}],
                system_prompt="You are a helpful assistant",
                response_model=TestAction
            )

            assert isinstance(result, CompletionResponse)
            assert isinstance(result.structured_output, TestAction)
            assert result.structured_output.query == "test query"
            assert result.completion == mock_response
