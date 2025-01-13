import logging
from typing import Optional, Dict, Any, Callable, List

from pydantic import BaseModel, Field

from moatless.agent.agent import ActionAgent
from moatless.artifacts.artifact import ArtifactChange, ArtifactHandler, Artifact
from moatless.artifacts.file import FileArtifact
from moatless.completion.model import Usage
from moatless.exceptions import RuntimeError
from moatless.node import Node
from moatless.schema import (
    Attachment,
    Message,
    UserMessage,
    ActionView,
    AssistantMessage,
)
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class Chat(BaseModel):
    current_node: Optional[Node] = Field(
        None, description="The root node of the chat sequence."
    )
    agent: ActionAgent = Field(..., description="Agent for generating responses.")
    artifact_handlers: List[ArtifactHandler] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the chat."
    )
    identifier_index: int = Field(0, description="")
    persist_path: Optional[str] = Field(
        None, description="Path to persist the chat sequence."
    )

    class Config:
        arbitrary_types_allowed = True

    def send_message(
        self, message: str, attachments: Optional[List[Attachment]] = None
    ) -> str:
        """Send a message with optional attachments and get a response."""

        if not self.current_node:
            workspace = Workspace(artifact_handlers=self.artifact_handlers)
        else:
            workspace = self.current_node.workspace.copy(deep=True)

        child_node = self._create_node(workspace, message, attachments)
        if self.current_node:
            child_node.set_parent(self.current_node)
        self.current_node = child_node

        try:
            self.agent.run(self.current_node)
            self.maybe_persist()
            return self.current_node.action or ""
        except RuntimeError as e:
            self.log(logger.error, f"Runtime error: {e.message}")
            return f"Error: {e.message}"

    def _create_node(
        self,
        workspace: Workspace,
        message: str,
        attachments: Optional[List[Attachment]] = None,
    ) -> Node:
        artifact_changes = []
        if attachments:
            for attachment in attachments:
                artifact = FileArtifact(
                    id=attachment.file_name,
                    type="file",
                    name=attachment.file_name,
                    file_path=attachment.file_name,
                    content=attachment.content,
                    mime_type=attachment.mime_type,
                )

                workspace.add_artifact(artifact)
                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=artifact.id,
                        change_type="added",
                        actor="user",
                    )
                )

        return Node(
            node_id=self._generate_unique_id(),
            workspace=workspace,
            artifact_changes=artifact_changes,
            user_message=message,
        )

    def get_messages(self) -> List[Message]:
        messages = []
        for node in self.current_node.get_trajectory():
            user_artifacts = [
                change.artifact_id
                for change in node.artifact_changes
                if change.actor == "user"
            ]

            if user_artifacts or node.user_message:
                messages.append(
                    UserMessage(content=node.user_message, artifact_ids=user_artifacts)
                )

            if node.action_steps or node.assistant_message:
                action_views = [
                    ActionView(name=action_step.action.name)
                    for action_step in node.action_steps
                ]
                messages.append(
                    AssistantMessage(
                        content=node.assistant_message, actions=action_views
                    )
                )

        return messages

    def get_artifacts(self) -> List[Artifact]:
        return self.current_node.workspace.artifacts

    def get_last_node(self) -> Node:
        """Get the last node in the chat sequence."""
        return self.root.get_all_nodes()[-1]

    def total_usage(self) -> Usage:
        """Calculate total token usage across all nodes."""
        total_usage = Usage()
        for node in self.root.get_all_nodes():
            total_usage += node.total_usage()
        return total_usage

    def maybe_persist(self):
        """Persist the chat state if a persist path is set."""
        if self.persist_path:
            self.persist(self.persist_path)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a dictionary representation of the Chat.

        Returns:
            Dict[str, Any]: A dictionary representation of the chat.
        """
        # Get all fields except the ones we'll handle separately
        data = {}
        data["metadata"] = self.metadata
        data["identifier_index"] = self.identifier_index
        data["agent"] = self.agent.model_dump(**kwargs)

        data["artifact_handlers"] = []
        for artifact_handler in self.artifact_handlers:
            data["artifact_handlers"].append(artifact_handler.model_dump(**kwargs))

        if self.current_node:
            data["nodes"] = self.current_node.dump_as_list(**kwargs)
            data["current_node_id"] = self.current_node.node_id

        return data

    def persist(self, file_path: str):
        """Persist the chat state to a file."""
        data = self.model_dump(exclude_none=True)
        with open(file_path, "w") as f:
            import json

            json.dump(data, f, indent=2)

    def _generate_unique_id(self) -> int:
        """Generate a unique ID for a new node."""
        self.identifier_index += 1
        return self.identifier_index

    def log(self, logger_fn: Callable, message: str, **kwargs):
        """Log a message with metadata."""
        metadata = {**self.metadata, **kwargs}
        metadata_str = " ".join(f"{k}: {str(v)[:20]}" for k, v in metadata.items())
        log_message = f"[{metadata_str}] {message}" if metadata else message
        logger_fn(log_message)
