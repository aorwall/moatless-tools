import json
from typing import List
from unittest.mock import Mock

from moatless.actions.find_class import FindClass
from moatless.actions.find_code_snippet import FindCodeSnippet
from moatless.actions.find_function import FindFunction
from moatless.actions.finish import Finish
from moatless.actions.reject import Reject
from moatless.actions.run_tests import RunTests
from moatless.actions.search_base import SearchBaseAction
from moatless.actions.semantic_search import SemanticSearch
from moatless.actions.string_replace import StringReplace
from moatless.actions.view_code import ViewCode
from moatless.agent.code_agent import CodingAgent
from moatless.completion import LLMResponseFormat
from moatless.completion.base import BaseCompletionModel
from moatless.index.code_index import CodeIndex
from moatless.repository.repository import InMemRepository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.runtime.runtime import TestResult


class MockCodeIndex(CodeIndex):
    def __init__(self):
        pass


class MockRuntimeEnvironment(RuntimeEnvironment):
    def run_tests(self, test_files: List[str] | None = None) -> list[TestResult]:
        return []



def test_dump_and_load_coding_agent():
    repository = InMemRepository()
    code_index = MockCodeIndex()
    runtime = MockRuntimeEnvironment()

    completion_model = BaseCompletionModel.create(
        response_format=LLMResponseFormat.TOOLS,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        model_api_key="dummy_key",
    )

    actions = [
        FindClass(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        FindFunction(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        FindCodeSnippet(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        SemanticSearch(repository=repository, code_index=code_index, completion_model=completion_model.clone()),
        ViewCode(repository=repository, completion_model=completion_model.clone()),
        StringReplace(repository=repository, completion_model=completion_model.clone()),
        RunTests(repository=repository, code_index=code_index, runtime=runtime),
        Finish(),
        Reject(),
    ]

    original_agent = CodingAgent(actions=actions, completion=completion_model, system_prompt="test")

    dumped_agent = json.dumps(original_agent.model_dump(), indent=2)
    print(dumped_agent)

    # Load the agent from JSON
    loaded_agent_data = json.loads(dumped_agent)
    loaded_agent = CodingAgent.model_validate(loaded_agent_data, repository=repository, code_index=code_index, runtime=runtime)

    # Manually set the dependencies after loading
    for action in loaded_agent.actions:
        if isinstance(action, SearchBaseAction):
            action._repository = repository
            action._code_index = code_index
        if hasattr(action, "runtime"):
            action.runtime = runtime

    # Assert that the number of actions is the same
    assert len(original_agent.actions) == len(loaded_agent.actions)

    # Assert that all action types are preserved
    original_action_types = set(type(action) for action in original_agent.actions)
    loaded_action_types = set(type(action) for action in loaded_agent.actions)
    assert original_action_types == loaded_action_types

    # Assert that the _action_map is correctly populated
    assert set(original_agent._action_map.keys()) == set(
        loaded_agent._action_map.keys()
    )

    # Assert that the completion model is preserved
    assert isinstance(loaded_agent.completion, BaseCompletionModel)
    assert loaded_agent.completion.model == original_agent.completion.model
    assert loaded_agent.completion.max_tokens == original_agent.completion.max_tokens
    assert loaded_agent.completion.temperature == original_agent.completion.temperature
    assert not loaded_agent.completion.model_api_key

    # Check if the dependencies are correctly set for each action
    for action in loaded_agent.actions:
        if hasattr(action, "_repository"):
            assert action._repository == repository
        if hasattr(action, "_code_index"):
            assert action._code_index == code_index
        if hasattr(action, "_runtime"):
            assert action._runtime == runtime

    for original_action, loaded_action in zip(
        original_agent.actions, loaded_agent.actions
    ):
        assert type(original_action) == type(loaded_action)
        assert original_action.name == loaded_action.name

