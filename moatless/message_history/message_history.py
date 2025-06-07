import logging
from typing import Any, List, Optional

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
    max_tokens_per_observation: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allowed in each observation, if exeeded the summary will be shown on previous messages",
    )
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
        if self.max_tokens:
            return await self._generate_token_limited_messages(previous_nodes, workspace)
        else:
            return await self._generate_unlimited_messages(previous_nodes, workspace)

    async def _generate_token_limited_messages(
        self, previous_nodes: List[Node], workspace: Workspace
    ) -> list[AllMessageValues]:
        """
        Generate messages based on token limits.

        This implementation will try to include the first message (user query)
        and as many of the recent messages as possible, prioritizing the most recent.
        The selection is dynamic based on the token limit.
        """
        # Find the last executed node (the last one with action steps)
        last_executed_idx = -1
        for i, node in enumerate(previous_nodes):
            if node.action_steps:
                last_executed_idx = i

        # First, generate all messages
        all_messages = []
        tool_idx = 0

        for i, node in enumerate(previous_nodes):
            is_last_executed_node = i == last_executed_idx
            node_messages, _ = await self._process_single_node(node, workspace, tool_idx, is_last_executed_node)

            # Update tool_idx for the next node
            for msg in node_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls")
                    if tool_calls is not None:
                        tool_idx += len(tool_calls)

            all_messages.extend(node_messages)

        # If we don't have a token limit or all messages fit, return them all
        total_tokens = sum(count_tokens(str(msg)) for msg in all_messages)
        if not self.max_tokens or total_tokens <= self.max_tokens:
            logger.info(f"Generated {len(all_messages)} messages with {total_tokens} tokens")
            return all_messages

        # We need to select messages based on the token limit
        # Always try to include the first message
        first_message = all_messages[0] if all_messages else None
        first_message_tokens = count_tokens(str(first_message)) if first_message else 0

        # If we can't fit even the first message, we need to truncate it somehow
        if first_message_tokens > self.max_tokens:
            logger.warning(f"First message exceeds token limit ({first_message_tokens} > {self.max_tokens})")
            # Return just the first message and we'll need to handle truncation elsewhere
            raise RuntimeError("First message exceeds token limit")

        # Start with the first message
        selected_messages = [first_message] if first_message else []
        remaining_tokens = self.max_tokens - first_message_tokens

        # Try to include recent messages, starting from the most recent
        for msg in reversed(all_messages[1:]):
            msg_tokens = count_tokens(str(msg))
            if msg_tokens <= remaining_tokens:
                # We can include this message
                selected_messages.insert(1, msg)  # Insert after first message
                remaining_tokens -= msg_tokens
            else:
                # This message doesn't fit
                continue

        # Sort messages to maintain conversation order
        selected_messages.sort(key=lambda msg: all_messages.index(msg))

        actual_tokens = sum(count_tokens(str(msg)) for msg in selected_messages)
        logger.info(
            f"Generated {len(selected_messages)} messages with {actual_tokens} tokens (limited by max_tokens={self.max_tokens})"
        )

        return selected_messages

    async def _process_single_node(
        self, node: Node, workspace: Workspace, tool_idx_start: int, is_last_node: bool = False
    ) -> tuple[list[AllMessageValues], int]:
        """
        Process a single node and return its messages and token count.

        Args:
            node: The node to process
            workspace: The workspace containing artifacts
            tool_idx_start: The starting index for tool calls
            is_last_node: Whether this is the most recent (last) node in the trajectory

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
            assistant_msg, tool_msgs, asst_tokens = await self._process_assistant_message(node, tool_idx, is_last_node)
            # Always add assistant message first if it has content or tool calls
            if assistant_msg and (assistant_msg.get("content") or assistant_msg.get("tool_calls")):
                messages.append(assistant_msg)
                # Then add tool responses
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
                            cache_control=None,
                        )
                    )
                    message_content.append(artifact.to_prompt_message_content())

        if node.user_message:
            message_content.append(ChatCompletionTextObject(type="text", text=node.user_message, cache_control=None))

        if message_content:
            user_message = ChatCompletionUserMessage(role="user", content=message_content, cache_control=None)
            tokens = count_tokens(str(message_content))
            return user_message, tokens

        return None, 0

    async def _process_assistant_message(
        self, node: Node, tool_idx_start: int, is_last_node: bool = False
    ) -> tuple[dict, list, int]:
        """
        Process the assistant message and tool calls, returning the messages and token count.

        Args:
            node: The node containing the assistant message and actions
            tool_idx_start: The starting index for tool calls
            is_last_node: Whether this is the most recent (last) node in the history

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
                observation_text = None

                if action_step.observation.message:
                    # For non-last nodes, use summary if message exceeds max_tokens_per_observation
                    if not is_last_node and self.max_tokens_per_observation:
                        message_tokens = count_tokens(action_step.observation.message)
                        if message_tokens > self.max_tokens_per_observation and action_step.observation.summary:
                            observation_text = action_step.observation.summary
                        else:
                            observation_text = action_step.observation.message
                    else:
                        # For the last (most recent) node, always use full message for all action steps
                        observation_text = action_step.observation.message

                    if observation_text:
                        message_content.append(
                            ChatCompletionTextObject(type="text", text=observation_text, cache_control=None)
                        )
                        tokens += count_tokens(observation_text)

                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": message_content,
                    }
                )

        # Build assistant message
        assistant_message: dict[str, Any] = {"role": "assistant"}
        assistant_content = []

        # Add thoughts if available (Claude specific)
        if node.thinking_blocks:
            assistant_message["thinking_blocks"] = node.thinking_blocks
        
        # Add assistant message if available
        if node.assistant_message:
            assistant_content.append(
                ChatCompletionTextObject(type="text", text=node.assistant_message, cache_control=None)
            )
            tokens += count_tokens(node.assistant_message)

        # Add content and tool calls to assistant message
        if assistant_content:
            assistant_message["content"] = assistant_content
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

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

        # Find the last executed node (the last one with action steps)
        last_executed_idx = -1
        for i, node in enumerate(previous_nodes):
            if node.action_steps:
                last_executed_idx = i

        # Process nodes in order (oldest to newest)
        for i, node in enumerate(previous_nodes):
            is_last_executed_node = i == last_executed_idx
            node_messages, node_tokens = await self._process_single_node(
                node, workspace, tool_idx, is_last_executed_node
            )

            # Update the tool index for tool calls in assistant messages
            for msg in node_messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls")
                    if tool_calls is not None:
                        tool_idx += len(tool_calls)

            messages.extend(node_messages)
            tokens += node_tokens

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages
