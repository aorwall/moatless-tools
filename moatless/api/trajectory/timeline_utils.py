from typing import List, Any

from moatless.actions.schema import Observation
from moatless.api.trajectory.schema import TimelineItemDTO, TimelineItemType
from moatless.node import Node, ActionStep


def create_user_message_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for user message."""
    if node.user_message:
        return TimelineItemDTO(
            label="User Message",
            type=TimelineItemType.USER_MESSAGE,
            content={"message": node.user_message}
        )
    return None

def create_user_artifact_items(node: Node) -> List[TimelineItemDTO]:
    """Create timeline items for user artifacts."""
    items: List[TimelineItemDTO] = []
    if node.artifact_changes:
        for artifact in node.artifact_changes:
            if artifact.actor == "user":
                items.append(TimelineItemDTO(
                    label="Artifact",
                    type=TimelineItemType.ARTIFACT,
                    content=artifact.model_dump()
                ))
    return items

def create_assistant_artifact_items(observation: Observation) -> List[TimelineItemDTO]:
    items: List[TimelineItemDTO] = []
    if observation and observation.artifact_changes:
        for artifact in observation.artifact_changes:
            if artifact.actor == "assistant":
                items.append(TimelineItemDTO(
                    label="Artifact",
                    type=TimelineItemType.ARTIFACT,
                    content=artifact.model_dump()
                ))
    return items

def create_assistant_message_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for assistant message."""
    if node.assistant_message:
        return TimelineItemDTO(
            label="Assistant Message",
            type=TimelineItemType.ASSISTANT_MESSAGE,
            content={"message": node.assistant_message}
        )
    return None

def create_completion_item(completion: Any, label: str = "Completion") -> TimelineItemDTO | None:
    """Create timeline item for a completion."""
    if completion and completion.usage:
        return TimelineItemDTO(
            label=label,
            type=TimelineItemType.COMPLETION,
            content={
                "usage": completion.usage.model_dump() if completion.usage else None,
                "input": completion.input,
                "response": completion.response
            }
        )
    return None

def create_thought_item(thoughts: str) -> TimelineItemDTO | None:
    """Create timeline item for thoughts."""
    if thoughts:
        return TimelineItemDTO(
            label="Thought",
            type=TimelineItemType.THOUGHT,
            content={"message": thoughts}
        )
    return None

def create_action_item(step: ActionStep) -> TimelineItemDTO | None:
    """Create timeline item for an action."""
    if step.action:
        return TimelineItemDTO(
            label=step.action.name,
            type=TimelineItemType.ACTION,
            content={
                **step.action.model_dump(exclude={"thoughts"}, exclude_unset=True, exclude_none=True),
                "errors": step.observation.properties.get("errors", []) if step.observation else [],
                "warnings": step.observation.properties.get("warnings", []) if step.observation else []
            }
        )
    return None

def create_observation_item(step: ActionStep) -> TimelineItemDTO | None:
    """Create timeline item for an observation."""
    if step.observation:
        return TimelineItemDTO(
            label="Observation",
            type=TimelineItemType.OBSERVATION,
            content=step.observation.model_dump()
        )
    return None

def create_error_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for an error."""
    if node.error:
        return TimelineItemDTO(
            label="Error",
            type=TimelineItemType.ERROR,
            content={"error": node.error}
        )
    return None

def create_workspace_item(node: Node) -> TimelineItemDTO | None:
    """Create timeline item for workspace."""
    if node.file_context and node.file_context.files:
        return TimelineItemDTO(
            label="Workspace",
            type=TimelineItemType.WORKSPACE,
            content=node.file_context.model_dump()
        )
    return None

def generate_timeline_items(node: Node) -> List[TimelineItemDTO]:
    """Generate timeline items for a node."""
    items: List[TimelineItemDTO] = []
    
    # Add user message if present
    if user_item := create_user_message_item(node):
        items.append(user_item)

    if user_artifact_items := create_user_artifact_items(node):
        items.extend(user_artifact_items)
    
    # Add assistant message if present
    if assistant_item := create_assistant_message_item(node):
        items.append(assistant_item)
    
    # Add action completion if present
    if "build_action" in node.completions:
        if completion_item := create_completion_item(node.completions["build_action"]):
            items.append(completion_item)
    
    for step in node.action_steps:
        if thoughts := getattr(step.action, "thoughts", None):
            if thought_item := create_thought_item(thoughts):
                items.append(thought_item)
        
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

    if workspace_item := create_workspace_item(node):
        items.append(workspace_item)

    return items