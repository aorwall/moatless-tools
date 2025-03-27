import logging
from typing import List, Optional

from moatless.completion.schema import (
    AllMessageValues,
    ChatCompletionTextObject,
    ChatCompletionUserMessage,
)
from moatless.message_history.base import BaseMemory
from moatless.node import Node
from moatless.utils.tokenizer import count_tokens
from moatless.workspace import Workspace
from pydantic import BaseModel, Field, PrivateAttr, model_serializer

logger = logging.getLogger(__name__)


class MessageHistoryGenerator(BaseMemory):
    """
    Generates chat completion messages from a node's history.

    This class converts the node trajectory into a sequence of messages that can be used
    for chat completion calls. It supports token limiting to maintain context within
    the model's token limits.
    """

    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens allowed in message history")
    include_file_context: bool = Field(default=True, description="Whether to include file context in messages")
    include_git_patch: bool = Field(default=True, description="Whether to include git patch in messages")
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    _workspace: Workspace | None = PrivateAttr(default=None)

    async def generate_messages(self, node: Node, workspace: Workspace) -> list[AllMessageValues]:  # type: ignore
        """
        Generate a list of messages from the node's trajectory.

        Args:
            node: The node to generate messages for
            workspace: The workspace containing artifacts

        Returns:
            A list of messages suitable for chat completion
        """
        previous_nodes = node.get_trajectory()

        # If we have a token limit, we need to process nodes from most recent to oldest
        # and include only those that fit within the limit
        if self.max_tokens is not None:
            return await self._generate_token_limited_messages(previous_nodes, workspace)
        else:
            return await self._generate_unlimited_messages(previous_nodes, workspace)

    async def _generate_token_limited_messages(
        self, previous_nodes: List[Node], workspace: Workspace
    ) -> list[AllMessageValues]:
        """
        Generate messages respecting the token limit by including only the most recent nodes.

        This method processes nodes from most recent to oldest, adding them until the token
        limit would be exceeded. This ensures the most relevant recent context is preserved.

        Args:
            previous_nodes: List of nodes in the trajectory (oldest to newest)
            workspace: The workspace containing artifacts

        Returns:
            A list of messages that fit within the token limit
        """
        if not self.max_tokens:
            return await self._generate_unlimited_messages(previous_nodes, workspace)

        # Process nodes in reverse order (most recent first)
        reversed_nodes = list(reversed(previous_nodes))

        # Temporary storage for messages and tokens
        temp_messages = []
        total_tokens = 0

        # Process each node starting from most recent
        for i, node in enumerate(reversed_nodes):
            node_messages, node_tokens = await self._process_single_node(node, workspace, len(reversed_nodes) - i)

            # If adding this node would exceed the token limit, stop
            if total_tokens + node_tokens > self.max_tokens and temp_messages:
                break

            # Add the messages from this node
            temp_messages = node_messages + temp_messages  # Prepend the older messages
            total_tokens += node_tokens

        # Verify the total token count is accurate
        actual_tokens = 0
        for message in temp_messages:
            message_str = str(message)
            actual_tokens += count_tokens(message_str)

        logger.info(
            f"Generated {len(temp_messages)} messages with {actual_tokens} tokens (limited by max_tokens={self.max_tokens})"
        )

        # If we're still over the limit, we need to remove messages from the beginning
        # This can happen because we're counting tokens at the node level, not message level
        if actual_tokens > self.max_tokens and len(temp_messages) > 1:
            # Start removing oldest messages until we're under the limit
            while actual_tokens > self.max_tokens and len(temp_messages) > 1:
                oldest_message = temp_messages.pop(0)
                oldest_tokens = count_tokens(str(oldest_message))
                actual_tokens -= oldest_tokens
                logger.info(
                    f"Removed oldest message ({oldest_tokens} tokens) to stay under limit. New total: {actual_tokens} tokens"
                )

        return temp_messages

    async def _process_single_node(
        self, node: Node, workspace: Workspace, tool_idx_start: int
    ) -> tuple[list[AllMessageValues], int]:
        """
        Process a single node and return its messages and token count.

        Args:
            node: The node to process
            workspace: The workspace containing artifacts
            tool_idx_start: The starting index for tool calls

        Returns:
            A tuple of (list of messages, token count)
        """
        messages = []
        tokens = 0
        tool_idx = tool_idx_start

        # Process user message
        user_message, user_tokens = await self._process_user_message(node, workspace)
        if user_message:
            messages.append(user_message)
            tokens += user_tokens

        # Process tool calls and assistant message
        if node.assistant_message or node.action_steps:
            assistant_msg, tool_msgs, asst_tokens = await self._process_assistant_message(node, tool_idx)
            if assistant_msg:
                messages.append(assistant_msg)
            messages.extend(tool_msgs)
            tokens += asst_tokens

        return messages, tokens

    async def _process_user_message(
        self, node: Node, workspace: Workspace
    ) -> tuple[Optional[ChatCompletionUserMessage], int]:
        """
        Process the user message part of a node and return the message and token count.

        Args:
            node: The node containing the user message
            workspace: The workspace containing artifacts

        Returns:
            A tuple of (user message or None, token count)
        """
        message_content = []
        tokens = 0

        if node.artifact_changes:
            for change in node.artifact_changes:
                artifact = workspace.get_artifact(change.artifact_type, change.artifact_id)
                if artifact:
                    message_content.append(
                        ChatCompletionTextObject(
                            type="text",
                            text=f"The {artifact.type} {artifact.id} was {change.change_type}",
                        )
                    )
                    message_content.append(artifact.to_prompt_message_content())

        if node.user_message:
            message_content.append(ChatCompletionTextObject(type="text", text=node.user_message))

        if message_content:
            user_message = ChatCompletionUserMessage(role="user", content=message_content)
            tokens = count_tokens(str(message_content))
            return user_message, tokens

        return None, 0

    async def _process_assistant_message(self, node: Node, tool_idx_start: int) -> tuple[dict, list, int]:
        """
        Process the assistant message and tool calls, returning the messages and token count.

        Args:
            node: The node containing the assistant message and actions
            tool_idx_start: The starting index for tool calls

        Returns:
            A tuple of (assistant message dict, list of tool responses, token count)
        """
        tokens = 0
        tool_idx = tool_idx_start
        tool_calls = []
        tool_responses = []

        # Process action steps (tool calls)
        for action_step in node.action_steps:
            tool_idx += 1
            tool_call_id = f"tool_{tool_idx}"

            exclude = None
            if not self.thoughts_in_action:
                exclude = {"thoughts"}

            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": action_step.action.name,
                        "arguments": action_step.action.model_dump_json(exclude=exclude),
                    },
                }
            )

            tokens += count_tokens(action_step.action.model_dump_json(exclude=exclude))

            # Process observation (tool response)
            if action_step.observation:
                message_content = []

                if action_step.observation.message:
                    message_content.append(ChatCompletionTextObject(type="text", text=action_step.observation.message))
                    tokens += count_tokens(action_step.observation.message)

                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": message_content,
                    }
                )

        # Build assistant message
        assistant_message = {"role": "assistant"}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        assistant_content = []
        if node.thoughts:
            assistant_content.extend(node.thoughts)

        if node.assistant_message:
            assistant_content.append(ChatCompletionTextObject(type="text", text=node.assistant_message))
            tokens += count_tokens(node.assistant_message)

        if assistant_content:
            assistant_message["content"] = assistant_content

        return assistant_message, tool_responses, tokens

    async def _generate_unlimited_messages(
        self, previous_nodes: List[Node], workspace: Workspace
    ) -> list[AllMessageValues]:
        """
        Generate messages without a token limit.

        Args:
            previous_nodes: List of nodes in the trajectory (oldest to newest)
            workspace: The workspace containing artifacts

        Returns:
            A list of all messages from the trajectory
        """
        messages = []
        tool_idx = 0
        tokens = 0

        # Process nodes in order (oldest to newest)
        for i, node in enumerate(previous_nodes):
            node_messages, node_tokens = await self._process_single_node(node, workspace, tool_idx)

            # Update the tool index for tool calls in assistant messages
            for msg in node_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_idx += len(msg["tool_calls"])

            messages.extend(node_messages)
            tokens += node_tokens

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages
