from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse, Usage, Message

from moatless.actions.string_replace import StringReplaceArgs
from moatless.actions.view_code import ViewCodeArgs
from moatless.actions.find_code_snippet import FindCodeSnippetArgs
from moatless.completion.base import LLMResponseFormat
from moatless.completion.react import ReActCompletionModel
from moatless.exceptions import CompletionRejectError


@pytest.fixture
def test_schema():
    return [StringReplaceArgs, ViewCodeArgs]


@pytest.fixture
def test_messages():
    return [{"role": "user", "content": "Update the code"}]


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM response with ReAct format content"""
    def _create_mock(content="", usage=None):
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


def test_prepare_schema_single(test_schema):
    """Test preparing single ReAct schema"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    model.initialize(response_schema=[test_schema[0]], system_prompt="Test prompt")
    assert len(model.response_schema) == 1
    assert model.response_schema[0] == test_schema[0]


def test_prepare_schema_multiple(test_schema):
    """Test preparing multiple ReAct schemas"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    assert len(model.response_schema) == 2
    assert model.response_schema[0] == test_schema[0]
    assert model.response_schema[1] == test_schema[1]


def test_validate_react_format_valid():
    """Test validating valid ReAct format"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    
    valid_response = """Action: StringReplace
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>"""
    
    # Should not raise any exceptions
    model._validate_react_format(valid_response)


def test_validate_react_format_with_thoughts():
    """Test validating ReAct format with thoughts"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    
    valid_response = """Thought: I should update the function to return True
Action: StringReplace
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>"""
    
    # Should not raise any exceptions
    model._validate_react_format(valid_response)


def test_validate_react_format_invalid_missing_action():
    """Test validating ReAct format with missing action"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    
    invalid_response = """Some random text without proper format"""
    
    with pytest.raises(ValueError, match="Response must have one 'Action:' section"):
        model._validate_react_format(invalid_response)


def test_validate_react_format_invalid_missing_thought():
    """Test validating ReAct format with missing thought when required"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    
    invalid_response = """Action: StringReplace
<path>test/file.py</path>
<old_str>old code</old_str>
<new_str>new code</new_str>"""
    
    with pytest.raises(ValueError, match="Response must have one 'Thought:' section"):
        model._validate_react_format(invalid_response)


def test_validate_react_format_invalid_order():
    """Test validating ReAct format with incorrect section order"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    
    invalid_response = """Action: StringReplace
<path>test/file.py</path>
<old_str>old code</old_str>
<new_str>new code</new_str>
Thought: This thought comes after action"""
    
    with pytest.raises(ValueError, match="Sections must be in order"):
        model._validate_react_format(invalid_response)


@patch("litellm.completion")
def test_validate_completion_valid_json(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating valid JSON completion"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    valid_response = """Thought: I need to view the code in the test file
Action: ViewCode
{
    "files": [
        {
            "file_path": "test/file.py",
            "start_line": 1,
            "end_line": 10,
            "span_ids": []
        }
    ]
}"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )
    
    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert isinstance(structured_outputs[0], ViewCodeArgs)
    assert structured_outputs[0].thoughts == "I need to view the code in the test file"
    assert len(structured_outputs[0].files) == 1
    assert structured_outputs[0].files[0].file_path == "test/file.py"
    assert text_response is None
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_valid_xml(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating valid XML completion"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    valid_response = """Thought: I should update the function implementation
Action: StringReplace
<path>test/file.py</path>
<old_str>
def old_function():
    pass
</old_str>
<new_str>
def new_function():
    return True
</new_str>"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )
    
    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert isinstance(structured_outputs[0], StringReplaceArgs)
    assert structured_outputs[0].path == "test/file.py"
    assert structured_outputs[0].old_str == "def old_function():\n    pass"
    assert structured_outputs[0].new_str == "def new_function():\n    return True"
    assert structured_outputs[0].thoughts == "I should update the function implementation"
    assert text_response is None
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_valid_xml_with_newlines(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating valid XML completion with newlines"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    valid_response = """Thought: I should update the function implementation
Action: StringReplace
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 18, "completion_tokens": 10, "total_tokens": 28}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )
    
    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert isinstance(structured_outputs[0], StringReplaceArgs)
    assert structured_outputs[0].path == "test/file.py"
    assert structured_outputs[0].old_str == "def old_function():\n    pass"
    assert structured_outputs[0].new_str == "def new_function():\n    return True"
    assert structured_outputs[0].thoughts == "I should update the function implementation"
    assert text_response is None
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_valid_xml_without_newlines(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating valid XML completion without newlines"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    valid_response = """Thought: I should update the function implementation
Action: StringReplace
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 18, "completion_tokens": 10, "total_tokens": 28}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )
    
    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert isinstance(structured_outputs[0], StringReplaceArgs)
    assert structured_outputs[0].path == "test/file.py"
    assert structured_outputs[0].old_str == "def old_function():\n    pass"
    assert structured_outputs[0].new_str == "def new_function():\n    return True"
    assert structured_outputs[0].thoughts == "I should update the function implementation"
    assert text_response is None
    assert flags == []


@patch("litellm.completion")
def test_validate_completion_invalid_action(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating completion with invalid action name"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    invalid_response = """Action: invalid_action
{
    "command": "test",
    "args": ["--flag"]
}"""
    
    mock_response = mock_litellm_response(
        invalid_response,
        usage={"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
    )
    
    with pytest.raises(CompletionRejectError, match="Unknown action"):
        model._validate_completion(
            completion_response=mock_response
        )


@patch("litellm.completion")
def test_validate_completion_invalid_json(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test validating completion with invalid JSON"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
    )
    model.initialize(response_schema=test_schema, system_prompt="Test prompt")
    
    invalid_response = """Action: ViewCode
{
    invalid json content
}"""
    
    mock_response = mock_litellm_response(
        invalid_response,
        usage={"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9}
    )
    
    with pytest.raises(CompletionRejectError):
        model._validate_completion(
            completion_response=mock_response
        )


@patch("litellm.completion")
def test_end_to_end_completion(mock_completion, mock_litellm_response, test_schema, test_messages):
    """Test complete ReAct completion flow"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=test_schema, system_prompt="Update the code")
    
    valid_response = """Thought: I should update the function implementation
Action: StringReplace
<path>test/file.py</path>
<old_str>def old_function():
    pass</old_str>
<new_str>def new_function():
    return True</new_str>"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}
    )
    mock_completion.return_value = mock_response
    
    result = model.create_completion(messages=test_messages)
    
    assert result.structured_outputs
    assert len(result.structured_outputs) == 1
    assert isinstance(result.structured_outputs[0], StringReplaceArgs)
    assert result.structured_outputs[0].path == "test/file.py"
    assert result.structured_outputs[0].old_str == "def old_function():\n    pass"
    assert result.structured_outputs[0].new_str == "def new_function():\n    return True"
    assert result.structured_outputs[0].thoughts == "I should update the function implementation" 


@patch("litellm.completion")
def test_validate_completion_find_code_snippet(mock_completion, mock_litellm_response):
    """Test validating FindCodeSnippet ReAct format response"""
    model = ReActCompletionModel(
        model="test",
        response_format=LLMResponseFormat.REACT,
        thoughts_in_action=True,
    )
    model.initialize(response_schema=[FindCodeSnippetArgs], system_prompt="Test prompt")
    
    valid_response = """Thought: I need to locate the exact line in `django/contrib/admin/options.py` where the regex pattern is generated without escaping the prefix. The code snippet provided in the task description is `pk_pattern = re.compile(r'{}-\d+-{}$'.format(prefix, self.model._meta.pk.name))`. I will use this exact snippet to find the relevant code.

Action: FindCodeSnippet
<code_snippet>
pk_pattern = re.compile(r'{}-\d+-{}$'.format(prefix, self.model._meta.pk.name))
</code_snippet>
<file_pattern>django/contrib/admin/options.py</file_pattern>"""
    
    mock_response = mock_litellm_response(
        valid_response,
        usage={"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40}
    )
    
    result = model._validate_completion(
        completion_response=mock_response
    )
    
    structured_outputs, text_response, flags = result
    assert structured_outputs
    assert len(structured_outputs) == 1
    assert isinstance(structured_outputs[0], FindCodeSnippetArgs)
    assert structured_outputs[0].file_pattern == "django/contrib/admin/options.py"
    assert structured_outputs[0].code_snippet == "pk_pattern = re.compile(r'{}-\d+-{}$'.format(prefix, self.model._meta.pk.name))"
    assert structured_outputs[0].thoughts == "I need to locate the exact line in `django/contrib/admin/options.py` where the regex pattern is generated without escaping the prefix. The code snippet provided in the task description is `pk_pattern = re.compile(r'{}-\d+-{}$'.format(prefix, self.model._meta.pk.name))`. I will use this exact snippet to find the relevant code."
    assert text_response is None
    assert flags == [] 