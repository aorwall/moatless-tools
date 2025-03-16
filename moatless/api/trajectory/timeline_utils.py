from typing import Any

from moatless.actions.schema import ActionArguments, Observation
from moatless.actions.think import ThinkArgs
from moatless.api.trajectory.schema import TimelineItemDTO, TimelineItemType
from moatless.node import ActionStep, Node


def create_user_message_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for user message."""
    if node.user_message:
        return TimelineItemDTO(
            label="User Message", type=TimelineItemType.USER_MESSAGE, content={"message": node.user_message}
        )
    return None


def create_user_artifact_items(node: Node) -> list[TimelineItemDTO]:
    """Create timeline items for user artifacts."""
    items: list[TimelineItemDTO] = []
    if node.artifact_changes:
        for artifact in node.artifact_changes:
            if artifact.actor == "user":
                items.append(
                    TimelineItemDTO(label="Artifact", type=TimelineItemType.ARTIFACT, content=artifact.model_dump())
                )
    return items


def create_assistant_artifact_items(observation: Observation) -> list[TimelineItemDTO]:
    items: list[TimelineItemDTO] = []
    if observation and observation.artifact_changes:
        for artifact in observation.artifact_changes:
            if artifact.actor == "assistant":
                items.append(
                    TimelineItemDTO(label="Artifact", type=TimelineItemType.ARTIFACT, content=artifact.model_dump())
                )
    return items


def create_assistant_message_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for assistant message."""
    if node.assistant_message:
        return TimelineItemDTO(
            label="Assistant Message",
            type=TimelineItemType.ASSISTANT_MESSAGE,
            content={"message": node.assistant_message},
        )
    return None


def create_completion_item(completion: Any, label: str = "Completion") -> TimelineItemDTO | None:
    """Create timeline item for a completion."""
    if completion and completion.usage:
        return TimelineItemDTO(
            label=label,
            type=TimelineItemType.COMPLETION,
            content={"usage": completion.usage.model_dump() if completion.usage else None},
        )
    return None


def create_thought_item(action: ActionArguments) -> TimelineItemDTO | None:
    """Create timeline item for thoughts."""
    if hasattr(action, "thoughts") and action.thoughts:
        return TimelineItemDTO(label="Thought", type=TimelineItemType.THOUGHT, content={"message": action.thoughts})
    elif isinstance(action, ThinkArgs):
        return TimelineItemDTO(label="Thought", type=TimelineItemType.THOUGHT, content={"message": action.thought})
    return None


def create_action_item(step: ActionStep) -> TimelineItemDTO | None:
    """Create timeline item for an action."""
    if step.action:
        if step.action.name == "Think":
            return None

        # Remove properties that are empty
        if step.action.name == "str_replace_editor" and hasattr(step.action, "command"):
            if step.action.command == "str_replace":
                action_name = "StringReplace"
            elif step.action.command == "create":
                action_name = "CreateFile"
            elif step.action.command == "view":
                action_name = "ViewCode"
            elif step.action.command == "insert":
                action_name = "InsertLines"
            else:
                action_name = "str_replace_editor"

            model_dump = step.action.model_dump(exclude={"thoughts", "command"}, exclude_unset=True, exclude_none=True)
        else:
            action_name = step.action.name
            model_dump = step.action.model_dump(exclude={"thoughts"}, exclude_unset=True, exclude_none=True)

        return TimelineItemDTO(
            label=action_name,
            type=TimelineItemType.ACTION,
            content={**model_dump},
        )
    return None


def create_observation_item(step: ActionStep) -> TimelineItemDTO | None:
    """Create timeline item for an observation."""
    if step.action.name == "Think":
        return None

    if step.observation:
        return TimelineItemDTO(
            label="Observation", type=TimelineItemType.OBSERVATION, content=step.observation.model_dump()
        )
    return None


def create_error_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for an error."""
    if node.error:
        return TimelineItemDTO(label="Error", type=TimelineItemType.ERROR, content={"error": node.error})
    return None


def create_workspace_files_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for updated files in workspace."""
    if node.file_context and node.file_context.files:
        # Get files that were edited or created
        updated_files = []
        for file in node.file_context.files:
            if file.patch:
                updated_files.append(
                    {
                        "file_path": file.file_path,
                        "is_new": file.is_new,
                        "has_patch": bool(file.patch),
                        "patch": file.patch,
                        "tokens": file.context_size() if hasattr(file, "context_size") else None,
                    }
                )

        if updated_files:
            return TimelineItemDTO(
                label="Updated Files",
                type=TimelineItemType.WORKSPACE_FILES,
                content={
                    "updatedFiles": updated_files,
                    "files": [f.model_dump() for f in node.file_context.files if f.was_edited or f.is_new],
                },
            )
    return None


def create_workspace_context_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for files in context."""
    if node.file_context and node.file_context.files:
        # Get all files in context that weren't edited
        context_files = []
        for file in node.file_context.files:
            if not file.was_edited and not file.is_new:
                context_files.append(
                    {
                        "file_path": file.file_path,
                        "tokens": file.context_size() if hasattr(file, "context_size") else None,
                        "spans": [span.model_dump() for span in file.spans] if hasattr(file, "spans") else [],
                    }
                )

        if context_files:
            return TimelineItemDTO(
                label="Files in Context",
                type=TimelineItemType.WORKSPACE_CONTEXT,
                content={
                    "files": context_files,
                    "max_tokens": node.file_context._max_tokens if hasattr(node.file_context, "_max_tokens") else None,
                },
            )
    return None


def create_workspace_tests_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for test results."""
    if node.file_context and hasattr(node.file_context, "_test_files") and node.file_context._test_files:
        test_files = []
        for file_path, test_file in node.file_context._test_files.items():
            if test_file.test_results:
                test_files.append(
                    {"file_path": file_path, "test_results": [result.model_dump() for result in test_file.test_results]}
                )

        if test_files:
            return TimelineItemDTO(
                label="Test Results", type=TimelineItemType.WORKSPACE_TESTS, content={"test_files": test_files}
            )
    return None


def create_reward_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for a reward."""
    if node.reward:
        return TimelineItemDTO(label="Reward", type=TimelineItemType.REWARD, content=node.reward.model_dump())
    return None


def generate_timeline_items(node: Node) -> list[TimelineItemDTO]:
    """Generate timeline items for a node."""
    items: list[TimelineItemDTO] = []

    if user_item := create_user_message_item(node):
        items.append(user_item)

    if user_artifact_items := create_user_artifact_items(node):
        items.extend(user_artifact_items)

    if "build_action" in node.completions:
        if completion_item := create_completion_item(node.completions["build_action"]):
            items.append(completion_item)

    if assistant_item := create_assistant_message_item(node):
        items.append(assistant_item)

    for step in node.action_steps:
        if thought_item := create_thought_item(step.action):
            items.append(thought_item)

        if isinstance(step.action, ThinkArgs):
            continue

        if action_item := create_action_item(step):
            items.append(action_item)

        if completion_item := create_completion_item(step.completion, "Action Completion"):
            items.append(completion_item)

        if assistant_artifact_items := create_assistant_artifact_items(step.observation):
            items.extend(assistant_artifact_items)

        if observation_item := create_observation_item(step):
            items.append(observation_item)

    if error_item := create_error_item(node):
        items.append(error_item)

    if workspace_files_item := create_workspace_files_item(node):
        items.append(workspace_files_item)

    if workspace_context_item := create_workspace_context_item(node):
        items.append(workspace_context_item)

    if workspace_tests_item := create_workspace_tests_item(node):
        items.append(workspace_tests_item)

    if reward_item := create_reward_item(node):
        items.append(reward_item)

    return items
