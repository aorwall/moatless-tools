import logging
from typing import List

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
    max_tokens: int = Field(default=20000, description="Maximum number of tokens allowed in message history")
    include_file_context: bool = Field(default=True, description="Whether to include file context in messages")
    include_git_patch: bool = Field(default=True, description="Whether to include git patch in messages")
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    _workspace: Workspace | None = PrivateAttr(default=None)

    async def generate_messages(self, node: Node, workspace: Workspace) -> list[AllMessageValues]:  # type: ignore
        previous_nodes = node.get_trajectory()

        messages = []
        tool_idx = 0
        tokens = 0

        for i, previous_node in enumerate(previous_nodes):
            # Handle user message
            message_content = []

            if previous_node.artifact_changes:
                for change in previous_node.artifact_changes:
                    artifact = workspace.get_artifact(change.artifact_type, change.artifact_id)
                    if artifact:
                        message_content.append(
                            ChatCompletionTextObject(
                                type="text",
                                text=f"The {artifact.type} {artifact.id} was {change.change_type}",
                            )
                        )
                        message_content.append(artifact.to_prompt_message_content())

            if previous_node.user_message:
                message_content.append(ChatCompletionTextObject(type="text", text=previous_node.user_message))

            if message_content:
                messages.append(ChatCompletionUserMessage(role="user", content=message_content))
                tokens += count_tokens(str(message_content))

            tool_calls = []
            tool_responses = []

            if not previous_node.assistant_message and not previous_node.action_steps:
                continue

            for action_step in previous_node.action_steps:
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

                message_content = []

                if action_step.observation:
                    if action_step.observation.message:
                        message_content.append(
                            ChatCompletionTextObject(type="text", text=action_step.observation.message)
                        )

                    # if action_step.observation.artifact_changes:
                    #    for change in action_step.observation.artifact_changes:
                    #        artifact = self._workspace.get_artifact(change.artifact_type, change.artifact_id)
                    #        if artifact:
                    #            message_content.append(
                    #                ChatCompletionTextObject(
                    #                    type="text",
                    #                    text=f"The {artifact.type} {artifact.id} was {change.change_type}",
                    #                )
                    #            )
                    #            message_content.append(artifact.to_prompt_message_content())

                    tool_responses.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": message_content,
                        }
                    )

                    # TODO: Count tokens for the artifact changes

                    tokens += count_tokens(action_step.observation.message)

            # TODO: Truncate on self.max_tokens!

            assistant_message = {"role": "assistant"}

            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            assistant_content = []

            if previous_node.thoughts:
                assistant_content.extend(previous_node.thoughts)

            if previous_node.assistant_message:
                assistant_content.append(ChatCompletionTextObject(type="text", text=previous_node.assistant_message))
                tokens += count_tokens(previous_node.assistant_message)

            if assistant_content:
                assistant_message["content"] = assistant_content

            messages.append(assistant_message)
            messages.extend(tool_responses)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages
