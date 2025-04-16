from moatless.actions.action import CompletionModelMixin
from moatless.completion.base import BaseCompletionModel
from moatless.value_function.base import BaseValueFunction
from moatless.node import Node, Reward
from typing import Optional, List, Any, Dict, cast
from pydantic import Field
from moatless.completion.schema import ResponseSchema, AllMessageValues

import logging
logger = logging.getLogger(__name__)


class ArtifactValidationResult(ResponseSchema):
    """
    Validates if the outcome of an action is correct based on artifact changes.
    """

    is_expected: bool = Field(..., description="Whether the absence of artifact changes is expected")
    reason_category: str = Field(
        ..., description="Categorization of reason: 'bug', 'incorrect_action', 'expected_behavior', 'other'"
    )
    explanation: str = Field(..., description="Explanation of the assessment")
    suggested_reward: int = Field(..., description="Suggested reward value, -100 to 100", ge=-100, le=100)


class ArtifactOutcomeValidator(BaseValueFunction, CompletionModelMixin):
    """
    A value function that verifies if the outcome from an action is correct by examining artifact changes.
    It classifies if the action was called correctly by the LLM and if the outcome is as expected.
    If the outcome is not as expected, it classifies whether it's due to a bug or something else.
    """

    # Configuration options
    check_empty_changes: bool = Field(default=True, description="Check when there are no artifact changes")
    check_action_types: List[str] = Field(
        default_factory=list, description="Specific action types to check. Empty list means check all actions."
    )
    ignore_action_types: List[str] = Field(
        default_factory=lambda: ["Reject", "Finish", "Think"],
        description="Action types to ignore even if they have no artifact changes. These are typically terminal actions like Reject or Finish.",
    )
    reward_for_bug: int = Field(default=-50, description="Reward value when a bug is detected", ge=-100, le=100)
    reward_for_incorrect_action: int = Field(
        default=-30, description="Reward value when incorrect action is detected", ge=-100, le=100
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_completion_model(self):
        """Initialize the completion model with the appropriate response schema"""
        if self.completion_model:
            self.completion_model = self.completion_model.clone()
            self.completion_model.initialize(
                response_schema=ArtifactValidationResult,
                system_prompt="""
You are an expert in evaluating whether an AI assistant's actions have the correct outcome.

Your task is to determine if the absence of artifact changes in the context is expected or a sign of a problem.

Artifact changes represent modifications to the context such as:
- New files added to the context
- Existing files updated or edited
- New information or content added to the agent's working context

When an agent performs an action, it should typically result in useful changes to the context/artifacts
if the action is meant to modify or add information. The absence of such changes might indicate a problem.

Categorize the assessment using these exact categories:

- 'bug': A technical error in the system where the observation doesn't match what would be expected 
    given the action and arguments. The implementation of the action doesn't work as intended.

- 'incorrect_action': The agent used the wrong action or incorrect arguments. If different arguments
    would have yielded the expected observation, it's an incorrect action. The agent made a mistake
    in its choice of arguments or action.

- 'expected_behavior': The absence of changes is normal for this action. The arguments are correct,
    and the implementation behaves properly. The lack of changes is part of the intended behavior.

- 'other': Another reason that doesn't fit the categories above.

Carefully analyze the action, observation, and context to make a detailed assessment.
Provide a detailed explanation and a reward value between -100 and 100.
""",
            )

    async def get_reward(self, node: Node) -> Optional[Reward]:
        """
        Determine if a reward should be given based on artifact changes.
        """
        if node.terminal:
            return None

        reward = await self._validate_artifact_changes(node)
        return reward

    async def _validate_artifact_changes(self, node: Node) -> Reward | None:
        """
        Validates if the absence of artifact changes is expected or not.
        """
        # Skip if no file context or parent context exists
        if not node.file_context or not node.parent or not node.parent.file_context:
            return None

        # Get artifact changes between parent and current node
        artifact_changes = node.file_context.get_artifact_changes(node.parent.file_context)

        # Skip if there are changes
        if artifact_changes:
            return None

        # Skip if the action is in the ignore list
        if node.action and node.action.name in self.ignore_action_types:
            return None

        # Skip if we have specific action types to check and this isn't one of them
        if self.check_action_types and node.action and node.action.name not in self.check_action_types:
            return None

        # If no completion model is available, use default behavior
        if not self.completion_model or not self.completion_model.initialized:
            return Reward(
                explanation="No artifact changes detected after the action", value=-20, tags=["no_artifact_changes"]
            )

        # Use LLM to evaluate if the absence of changes is expected
        return await self._evaluate_with_llm(node, artifact_changes)

    async def _evaluate_with_llm(self, node: Node, artifact_changes: list) -> Reward | None:
        """
        Use LLM to evaluate if the absence of changes is expected or a problem.
        """
        # Create context information for the LLM
        action_info = f"Action: {node.action.name}" if node.action else "No action"
        action_args = node.action.model_dump() if node.action else {}

        # Get observation text safely, preferring message attribute if available
        observation_text = "No observation"
        if node.observation:
            if hasattr(node.observation, "message"):
                observation_text = str(node.observation.message)
            else:
                observation_text = str(node.observation)

        # Check for files in context as a proxy for artifacts
        has_files = False
        if node.file_context and hasattr(node.file_context, "files"):
            has_files = bool(node.file_context.files)

        files_info = ""
        if node.file_context and hasattr(node.file_context, "file_paths"):
            files_info = f"Files in context: {', '.join(node.file_context.file_paths) if node.file_context.file_paths else 'None'}"

        artifact_changes_description = """
Artifact changes represent modifications to the context such as:
- New files added to the context
- Existing files updated or edited
- Files removed from the context
- New information or content added to the agent's working context

The absence of such changes might indicate a problem or might be expected behavior
depending on the action type and arguments.
"""

        context_status = "Has files in context" if has_files else "No files in context"

        # Create user message content
        message_content = f"""
Evaluate whether the absence of artifact changes after the following action is expected or indicates a problem:

{action_info}

Action arguments: {action_args}

Observation result: 
{observation_text}

Current context status: {context_status}
{files_info}

{artifact_changes_description}

There were NO changes to the artifacts/context after this action was executed.

Evaluate whether this is expected or a sign of a problem. Remember:
- 'bug' means the action implementation has a technical problem and doesn't work as intended
- 'incorrect_action' means the agent used the wrong action or arguments
- 'expected_behavior' means the absence of changes is normal for this action with these arguments
"""

        # Create message for the LLM using the correct typing
        messages: List[AllMessageValues] = [{"role": "user", "content": message_content}]

        # Call the completion model if available
        if not self.completion_model:
            raise ValueError("No completion model available for artifact outcome validation")

        # Call the completion model
        response = await self.completion_model.create_completion(messages=messages)

        if not response.structured_outputs:
            return Reward(
                explanation="Could not evaluate absence of artifact changes", value=-10, tags=["evaluation_failed"]
            )

        # Extract the validation result
        result = response.structured_outputs[0]
        if not isinstance(result, ArtifactValidationResult):
            return Reward(explanation="Invalid evaluation response", value=-10, tags=["invalid_response"])

        # Map reason category to reward value and tags
        reward_value = result.suggested_reward
        tags = ["artifact_validation", result.reason_category]

        if result.reason_category == "bug":
            reward_value = self.reward_for_bug
            tags.append("bug_detected")
        elif result.reason_category == "incorrect_action":
            reward_value = self.reward_for_incorrect_action
            tags.append("incorrect_action")

        return Reward(
            explanation=result.explanation, value=reward_value, tags=tags, completion=response.completion_invocation
        )


    @classmethod
    def model_validate(cls, data: Any, **kwargs) -> "ArtifactOutcomeValidator":
        """
        Validate the model configuration.
        """
        
        if isinstance(data, dict):
            if "completion_model" in data:
                data["completion_model"] = BaseCompletionModel.from_dict(data["completion_model"])
                
        return super().model_validate(data, **kwargs)
    
    def model_dump(self, **kwargs) -> dict:
        """
        Dump the model configuration.
        """
        data = super().model_dump(**kwargs)
        if self.completion_model:
            data["completion_model"] = self.completion_model.model_dump()
        return data