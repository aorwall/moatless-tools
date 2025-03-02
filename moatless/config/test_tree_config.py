import json
from pathlib import Path

import pytest
from swesearch.discriminator import AgentDiscriminator
from swesearch.feedback.diff_agent import DiffAgent
from swesearch.selector.selector import DepthFirstSelector
from swesearch.value_function.coding import CodingValueFunction

from moatless.completion.base import BaseCompletionModel
from moatless.flow.manager import TreeConfig, TreeConfigManager


@pytest.fixture
def test_config_path(tmp_path):
    return tmp_path / "test_trees.json"


@pytest.fixture
def test_completion_model():
    return BaseCompletionModel(model_id="test-model", api_key="test-key")


def test_tree_config_save_load(test_config_path, test_completion_model):
    # Create components
    selector = DepthFirstSelector(max_trajectory_depth=25)
    feedback_generator = DiffAgent(finished_nodes_only=True, max_trajectory_depth=20)
    discriminator = AgentDiscriminator(completion=test_completion_model, n_agents=3, n_rounds=2)

    value_function = CodingValueFunction()

    # Create config
    config = TreeConfig(
        config_id="test_tree",
        description="Test tree configuration",
        model_id="gpt-4",
        agent_id="test-agent",
        max_expansions=5,
        max_iterations=500,
        max_depth=25,
        reward_threshold=101,
        max_finished_nodes=20,
        selector=selector,
        feedback_generator=feedback_generator,
        discriminator=discriminator,
    )

    # Create manager with custom path
    class TestTreeConfigManager(TreeConfigManager):
        def _get_config_path(self) -> Path:
            return test_config_path

    # Save config
    manager = TestTreeConfigManager()
    manager.create_config(config)

    # Verify file was created and contains expected data
    assert test_config_path.exists()

    with open(test_config_path) as f:
        saved_configs = json.load(f)
    assert len(saved_configs) == 1
    saved_config = saved_configs[0]

    # Verify key fields
    assert saved_config["config_id"] == "test_tree"
    assert saved_config["model_id"] == "gpt-4"
    assert saved_config["agent_id"] == "test-agent"
    assert saved_config["max_expansions"] == 5
    assert saved_config["max_iterations"] == 500
    assert saved_config["max_depth"] == 25
    assert saved_config["reward_threshold"] == 101
    assert saved_config["max_finished_nodes"] == 20

    # Verify components were saved with their class info
    assert saved_config["selector"]["selector_class"] == "moatless.selector.selector.DepthFirstSelector"
    assert saved_config["selector"]["max_trajectory_depth"] == 25

    assert saved_config["feedback_generator"]["feedback_class"] == "moatless.feedback.diff_agent.DiffAgent"
    assert saved_config["feedback_generator"]["finished_nodes_only"] is True
    assert saved_config["feedback_generator"]["max_trajectory_depth"] == 20

    assert saved_config["discriminator"]["completion"]["model_id"] == "test-model"

    # Load config back
    loaded_config = manager.get_tree_config("test_tree")

    # Verify loaded config matches original
    assert loaded_config.config_id == config.config_id
    assert loaded_config.model_id == config.model_id
    assert loaded_config.agent_id == config.agent_id
    assert loaded_config.max_expansions == config.max_expansions
    assert loaded_config.max_iterations == config.max_iterations
    assert loaded_config.max_depth == config.max_depth
    assert loaded_config.reward_threshold == config.reward_threshold
    assert loaded_config.max_finished_nodes == config.max_finished_nodes

    # Verify components were loaded correctly
    assert isinstance(loaded_config.selector, DepthFirstSelector)
    assert loaded_config.selector.max_trajectory_depth == 25

    assert isinstance(loaded_config.feedback_generator, DiffAgent)
    assert loaded_config.feedback_generator.finished_nodes_only is True
    assert loaded_config.feedback_generator.max_trajectory_depth == 20

    assert isinstance(loaded_config.discriminator, AgentDiscriminator)
    assert loaded_config.discriminator.completion.model_id == "test-model"
