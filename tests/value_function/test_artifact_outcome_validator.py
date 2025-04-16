import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from moatless.value_function.artifact_outcome_validator import ArtifactOutcomeValidator, ArtifactValidationResult
from moatless.node import Node, Reward
from moatless.completion.base import CompletionResponse
from moatless.artifacts.artifact import ArtifactChange
from moatless.completion.stats import CompletionInvocation
from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.reject import Reject, RejectArgs


# Mock classes for testing
class MockFileContext:
    def __init__(self, artifacts=None):
        self.artifacts = artifacts or []
        self.files = []
        self.file_paths = []

    def get_artifact_changes(self, other_context):
        return []  # Empty changes for our tests


class MockAction:
    def __init__(self, name="test_action"):
        self.name = name

    def model_dump(self):
        return {"name": self.name}


class MockObservation:
    def __init__(self, message="Test observation message"):
        self.message = message

    def __str__(self):
        return self.message


@pytest.fixture
def mock_node():
    node = MagicMock(spec=Node)
    node.terminal = False
    node.file_context = MockFileContext()
    node.parent = MagicMock()
    node.parent.file_context = MockFileContext()
    node.action = MockAction()
    node.observation = MockObservation()
    return node


@pytest.fixture
def mock_node_with_changes():
    node = MagicMock(spec=Node)
    node.terminal = False
    node.file_context = MockFileContext()
    node.parent = MagicMock()
    node.parent.file_context = MockFileContext()

    # Mock that there are artifact changes
    node.file_context.get_artifact_changes = MagicMock(
        return_value=[
            ArtifactChange(
                artifact_id="test.py", artifact_type="file", change_type="added", properties={}, actor="assistant"
            )
        ]
    )

    node.action = MockAction()
    node.observation = MockObservation()
    return node


@pytest.fixture
def mock_completion_invocation():
    return MagicMock(spec=CompletionInvocation)


@pytest.fixture
def mock_completion_response_expected(mock_completion_invocation):
    response = MagicMock(spec=CompletionResponse)
    validation_result = ArtifactValidationResult(
        is_expected=True,
        reason_category="expected_behavior",
        explanation="This is expected because the action was just a query.",
        suggested_reward=0,
    )
    response.structured_outputs = [validation_result]
    response.completion_invocation = mock_completion_invocation
    return response


@pytest.fixture
def mock_completion_response_bug(mock_completion_invocation):
    response = MagicMock(spec=CompletionResponse)
    validation_result = ArtifactValidationResult(
        is_expected=False, reason_category="bug", explanation="This is a bug in the system.", suggested_reward=-50
    )
    response.structured_outputs = [validation_result]
    response.completion_invocation = mock_completion_invocation
    return response


@pytest.fixture
def mock_completion_model(mock_completion_response_expected):
    model = MagicMock()
    model.initialized = True
    model.create_completion = AsyncMock(return_value=mock_completion_response_expected)
    return model


class TestArtifactOutcomeValidator:
    @pytest.mark.asyncio
    async def test_skip_terminal_node(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator()
        mock_node.terminal = True

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_skip_when_changes_exist(self, mock_node_with_changes):
        # Arrange
        validator = ArtifactOutcomeValidator()

        # Act
        result = await validator.get_reward(mock_node_with_changes)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_default_reward_no_completion_model(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator()
        validator.completion_model = None

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is not None
        assert result.value == -20
        assert result.tags is not None
        assert "no_artifact_changes" in result.tags
        assert result.explanation is not None
        assert "No artifact changes detected" in result.explanation

    @pytest.mark.asyncio
    async def test_filtered_action_types(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator(check_action_types=["other_action"])
        mock_node.action.name = "test_action"  # Not in check_action_types

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_ignored_action_types(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator(ignore_action_types=["test_action", "another_action"])
        mock_node.action.name = "test_action"  # In ignore_action_types

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is None

        # Test with action not in ignore list
        mock_node.action.name = "not_ignored_action"
        result = await validator.get_reward(mock_node)

        # Should proceed with evaluation
        assert result is not None

    @pytest.mark.asyncio
    async def test_real_finish_action_name_ignored(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator()  # Default ignores Finish

        # Instead of setting the actual action object, we'll just use its name
        mock_node.action = MockAction(name="Finish")

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_real_reject_action_name_ignored(self, mock_node):
        # Arrange
        validator = ArtifactOutcomeValidator()  # Default ignores Reject

        # Instead of setting the actual action object, we'll just use its name
        mock_node.action = MockAction(name="Reject")

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_evaluation_expected(self, mock_node, mock_completion_model, mock_completion_response_expected):
        # Arrange
        validator = ArtifactOutcomeValidator()
        validator.completion_model = mock_completion_model

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is not None
        assert result.tags is not None
        assert "expected_behavior" in result.tags
        assert result.explanation == "This is expected because the action was just a query."

    @pytest.mark.asyncio
    async def test_llm_evaluation_bug(self, mock_node, mock_completion_model, mock_completion_response_bug):
        # Arrange
        validator = ArtifactOutcomeValidator(reward_for_bug=-75)
        validator.completion_model = mock_completion_model
        # Make sure completion_model is not None before setting attribute
        assert validator.completion_model is not None
        validator.completion_model.create_completion = AsyncMock(return_value=mock_completion_response_bug)

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is not None
        assert result.tags is not None
        assert "bug" in result.tags
        assert "bug_detected" in result.tags
        assert result.value == -75  # Should use the configured value
        assert result.explanation == "This is a bug in the system."

    @pytest.mark.asyncio
    async def test_llm_evaluation_no_structured_output(
        self, mock_node, mock_completion_model, mock_completion_invocation
    ):
        # Arrange
        validator = ArtifactOutcomeValidator()
        validator.completion_model = mock_completion_model
        # Make sure completion_model is not None before setting attribute
        assert validator.completion_model is not None
        mock_empty_response = MagicMock(spec=CompletionResponse)
        mock_empty_response.structured_outputs = []
        mock_empty_response.completion_invocation = mock_completion_invocation
        validator.completion_model.create_completion = AsyncMock(return_value=mock_empty_response)

        # Act
        result = await validator.get_reward(mock_node)

        # Assert
        assert result is not None
        assert result.value == -10
        assert result.tags is not None
        assert "evaluation_failed" in result.tags
