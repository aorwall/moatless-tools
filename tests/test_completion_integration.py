import pytest

from moatless.actions.finish import FinishArgs
from moatless.actions.reject import RejectArgs
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import AssistantMessage, UserMessage, Usage
from tests.conftest import TEST_MODELS


@pytest.mark.parametrize("model", TEST_MODELS)
@pytest.mark.llm_integration
def test_create_completion(model):
    completion_model = CompletionModel(model=model, temperature=0.0)

    actions = [RejectArgs, FinishArgs]

    messages = [
        UserMessage(content="Please complete this task."),
        AssistantMessage(content="Certainly! I'll complete the task for you."),
        UserMessage(content="Great, please finish the task now."),
    ]

    system_prompt = (
        "You are an AI assistant. When asked to finish a task, use the Finish action."
    )

    action_args, completion = completion_model.create_completion(
        messages=messages, system_prompt=system_prompt, actions=actions
    )

    assert isinstance(action_args, FinishArgs)
    assert hasattr(action_args, "finish_reason")
    assert hasattr(action_args, "scratch_pad")

    assert completion.model == model
    assert completion.usage is not None
    assert isinstance(completion.usage, Usage)
    assert completion.usage.completion_tokens > 0
    assert completion.usage.prompt_tokens > 0


@pytest.mark.parametrize("model", TEST_MODELS)
@pytest.mark.llm_integration
def test_create_text_completion(model):
    completion_model = CompletionModel(model=model, temperature=0.0)

    messages = [
        UserMessage(
            content="What is the capital of France? Respond with only the name of the capital."
        ),
    ]

    system_prompt = "You are an AI assistant. Provide informative answers to questions."

    response, completion = completion_model.create_text_completion(
        messages=messages,
        system_prompt=system_prompt,
    )

    assert "Paris" in response

    assert completion.model == model
    assert completion.usage is not None
    assert isinstance(completion.usage, Usage)
    assert completion.usage.completion_tokens > 0
    assert completion.usage.prompt_tokens > 0

def test_create_completion_qwen_coder():
    completion_model = CompletionModel(model="openai/Qwen/Qwen2.5-Coder-32B-Instruct", temperature=0.0)
    completion_model.response_format = LLMResponseFormat.TOOLS

    actions = [RejectArgs, FinishArgs]

    messages = [
        UserMessage(content="Please complete this task."),
        AssistantMessage(content="Certainly! I'll complete the task for you."),
        UserMessage(content="Great, please finish the task now."),
    ]

    system_prompt = (
        "You are an AI assistant. When asked to finish a task, use the Finish action."
    )

    action_args, completion = completion_model.create_completion(
        messages=messages, system_prompt=system_prompt, actions=actions
    )

    assert isinstance(action_args, FinishArgs)
    assert hasattr(action_args, "finish_reason")
    assert hasattr(action_args, "scratch_pad")

    assert completion.usage is not None
    assert isinstance(completion.usage, Usage)
    assert completion.usage.completion_tokens > 0
    assert completion.usage.prompt_tokens > 0