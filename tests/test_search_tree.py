import json

from moatless.actions.code_change import RequestCodeChange
from moatless.actions.find_class import FindClass
from moatless.actions.run_tests import RunTests
from moatless.agent.agent import ActionAgent
from moatless.completion.completion import CompletionModel
from moatless.discriminator import MeanAwardDiscriminator
from moatless.feedback import FeedbackGenerator
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository
from moatless.runtime.runtime import NoEnvironment
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector
from moatless.value_function.base import ValueFunction


def test_search_tree_dump_and_load():
    repository = FileRepository(repo_path="/tmp")
    code_index = CodeIndex(repository)
    runtime = NoEnvironment()

    original_model_name = "gpt-4"
    completion_model = CompletionModel(model=original_model_name)

    run_tests = RunTests(repository=repository, code_index=code_index, runtime=runtime)
    code_change = RequestCodeChange(repository=repository, completion_model=completion_model)
    find_class = FindClass(repository=repository, code_index=code_index)

    search_tree = SearchTree.create(
        message="Test input",
        selector=BestFirstSelector(),
        agent=ActionAgent(
            actions=[code_change, run_tests, find_class],
            completion=completion_model,
        ),
        value_function=ValueFunction(
            completion=completion_model
        ),
        feedback_generator=FeedbackGenerator(),
        discriminator=MeanAwardDiscriminator(),
        max_expansions=1,
        max_iterations=25,
    )

    dumped_data = search_tree.model_dump(exclude_none=True)
    print(json.dumps(dumped_data, indent=2))

    loaded_tree = SearchTree.from_dict(
        dumped_data, repository=repository, runtime=runtime, code_index=code_index
    )

    assert loaded_tree.agent._completion.model == original_model_name
    assert loaded_tree.value_function._completion.model == original_model_name

    assert loaded_tree.root.node_id == search_tree.root.node_id
    assert isinstance(loaded_tree.selector, BestFirstSelector)
    assert isinstance(loaded_tree.agent, ActionAgent)
    assert isinstance(loaded_tree.value_function, ValueFunction)
    assert isinstance(loaded_tree.feedback_generator, FeedbackGenerator)
    assert isinstance(loaded_tree.discriminator, MeanAwardDiscriminator)

    assert loaded_tree.max_expansions == search_tree.max_expansions
    assert loaded_tree.max_iterations == search_tree.max_iterations
    assert loaded_tree.max_cost == search_tree.max_cost
    assert loaded_tree.min_finished_nodes == search_tree.min_finished_nodes
    assert loaded_tree.reward_threshold == search_tree.reward_threshold
    assert loaded_tree.max_depth == search_tree.max_depth
    assert loaded_tree.metadata == search_tree.metadata

    assert loaded_tree.agent.actions[0]._completion_model
    assert loaded_tree.agent.actions[0]._repository == repository

    assert loaded_tree.agent.actions[1]._repository == repository
    assert loaded_tree.agent.actions[1]._code_index == code_index
    assert loaded_tree.agent.actions[1]._runtime == runtime

    assert loaded_tree.agent.actions[2]._repository == repository
    assert loaded_tree.agent.actions[2]._code_index == code_index
