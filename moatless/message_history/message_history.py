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
        if self.max_tokens:
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
        The first user message and the most recent node's messages are always included.

        Args:
            previous_nodes: List of nodes in the trajectory (oldest to newest)
            workspace: The workspace containing artifacts

        Returns:
            A list of messages that fit within the token limit
        """
        if not self.max_tokens:
            return await self._generate_unlimited_messages(previous_nodes, workspace)

        # Generate all messages to identify the ones we need
        all_messages = await self._generate_unlimited_messages(previous_nodes, workspace)

        # If we have 5 or fewer messages already, return them all
        if len(all_messages) <= 5:
            return all_messages

        # Get the first user message
        first_user_message = next(
            (msg for msg in all_messages if isinstance(msg, dict) and msg.get("role") == "user"), None
        )

        # Get the last 4 messages
        last_four_messages = all_messages[-4:]

        # Combine first message and last 4 messages
        result_messages = []
        if first_user_message:
            result_messages.append(first_user_message)
        result_messages.extend(last_four_messages)

        # Calculate total tokens
        actual_tokens = 0
        for message in result_messages:
            message_str = str(message)
            actual_tokens += count_tokens(message_str)

        # Check if we're within the token limit
        if actual_tokens <= self.max_tokens:
            logger.info(
                f"Generated {len(result_messages)} messages with {actual_tokens} tokens (limited by max_tokens={self.max_tokens})"
            )
            return result_messages

        # If we exceed the token limit, revert to the previous algorithm
        return await self._generate_adaptive_token_limited_messages(previous_nodes, workspace)

    async def _generate_adaptive_token_limited_messages(
        self, previous_nodes: List[Node], workspace: Workspace
    ) -> list[AllMessageValues]:
        """
        Adaptively generate messages based on token limits.

        Fallback implementation when the specific first+last4 approach doesn't fit in token limit.
        """
        # First, process the initial node to get the user message
        first_node = previous_nodes[0]
        first_messages, first_tokens = await self._process_single_node(first_node, workspace, 0)
        first_user_message = next(
            (msg for msg in first_messages if isinstance(msg, dict) and msg["role"] == "user"), None
        )
        first_user_tokens = count_tokens(str(first_user_message)) if first_user_message else 0

        # Process the most recent node
        last_node = previous_nodes[-1]
        last_messages, last_tokens = await self._process_single_node(last_node, workspace, len(previous_nodes))

        # Find the assistant message and its tool responses in the last node
        last_assistant_idx = next(
            (i for i, msg in enumerate(last_messages) if msg.get("role") == "assistant"),
            -1,
        )

        # Start with just the first user message
        temp_messages = []
        if first_user_message:
            temp_messages.append(first_user_message)

        # Add messages from the last node, ensuring we include the assistant message and its tool responses
        total_tokens = first_user_tokens
        if last_assistant_idx >= 0:
            # Add the assistant message
            assistant_msg = last_messages[last_assistant_idx]
            assistant_tokens = count_tokens(str(assistant_msg))

            # Add any tool responses that follow
            tool_responses = [msg for msg in last_messages[last_assistant_idx + 1 :] if msg.get("role") == "tool"]

            # Calculate total tokens needed for assistant message and tool responses
            total_needed = assistant_tokens + sum(count_tokens(str(msg)) for msg in tool_responses)

            # If we can fit both assistant message and all tool responses, add them
            if total_tokens + total_needed <= self.max_tokens:
                temp_messages.append(assistant_msg)
                temp_messages.extend(tool_responses)
                total_tokens += total_needed
            else:
                # Try to fit as many of the tool responses as possible, starting from the end
                # Always include the assistant message if possible
                remaining_tokens = self.max_tokens - total_tokens - assistant_tokens
                included_tool_responses = []

                # Work backward from the end to include the most recent tool responses first
                for tool_msg in reversed(tool_responses):
                    tool_tokens = count_tokens(str(tool_msg))
                    if remaining_tokens >= tool_tokens:
                        included_tool_responses.insert(0, tool_msg)  # Insert at beginning to maintain order
                        remaining_tokens -= tool_tokens
                    else:
                        break

                # Only add the assistant message if we can include at least one tool response
                # or if it fits on its own
                if included_tool_responses or (total_tokens + assistant_tokens <= self.max_tokens):
                    temp_messages.append(assistant_msg)
                    total_tokens += assistant_tokens

                    # Add the tool responses we could fit
                    temp_messages.extend(included_tool_responses)
                    total_tokens += self.max_tokens - remaining_tokens - total_tokens + assistant_tokens

        # Calculate total tokens
        actual_tokens = 0
        for message in temp_messages:
            message_str = str(message)
            actual_tokens += count_tokens(message_str)

        logger.info(
            f"Generated {len(temp_messages)} messages with {actual_tokens} tokens (limited by max_tokens={self.max_tokens})"
        )

        # Verify message order
        for i in range(len(temp_messages) - 1):
            if temp_messages[i].get("role") == "user":
                assert temp_messages[i + 1].get("role") == "assistant", "Assistant message must follow user message"

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
        assistant_content = []

        # Add thoughts if available
        if node.thoughts:
            assistant_content.extend(node.thoughts)

        # Add assistant message if available
        if node.assistant_message:
            assistant_content.append(ChatCompletionTextObject(type="text", text=node.assistant_message))
            tokens += count_tokens(node.assistant_message)

        # If we have tool calls but no content, add a default message
        if tool_calls and not assistant_content:
            default_message = ChatCompletionTextObject(type="text", text="Executing actions...")
            assistant_content.append(default_message)
            tokens += count_tokens("Executing actions...")

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
