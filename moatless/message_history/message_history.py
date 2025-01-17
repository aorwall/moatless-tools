import logging
from typing import List

from pydantic import BaseModel, Field, model_serializer

from moatless.completion.schema import ChatCompletionUserMessage, AllMessageValues
from moatless.node import Node
from moatless.schema import MessageHistoryType
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class MessageHistoryGenerator(BaseModel):
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Type of message history to generate",
    )
    max_tokens: int = Field(default=20000, description="Maximum number of tokens allowed in message history")
    include_file_context: bool = Field(default=True, description="Whether to include file context in messages")
    include_git_patch: bool = Field(default=True, description="Whether to include git patch in messages")
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    @model_serializer
    def serialize_model(self) -> dict:
        return {
            "message_history_type": self.message_history_type.value,
            "max_tokens": self.max_tokens,
            "include_file_context": self.include_file_context,
            "include_git_patch": self.include_git_patch,
            "thoughts_in_action": self.thoughts_in_action,
        }

    def generate_messages(self, node: Node) -> List[AllMessageValues]:  # type: ignore
        previous_nodes = node.get_trajectory()

        messages = []
        tool_idx = 0
        tokens = 0

        for i, previous_node in enumerate(previous_nodes):
            # Handle user message
            if previous_node.user_message:
                message_content = [{"type": "text", "text": previous_node.user_message}]

                if previous_node.artifact_changes:
                    for change in previous_node.artifact_changes:
                        artifact = previous_node.workspace.get_artifact_by_id(change.artifact_id)
                        if artifact:
                            message = f"{artifact.type} artifact: {artifact.id}"
                            message_content.append({"type": "text", "text": message})
                            message_content.append(artifact.to_prompt_format())

                messages.append(ChatCompletionUserMessage(role="user", content=message_content))
                tokens += count_tokens(previous_node.user_message)

            tool_calls = []
            tool_responses = []

            if previous_node.feedback_data:
                messages.append(ChatCompletionUserMessage(role="user", content=previous_node.feedback_data.feedback))

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

                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": action_step.observation.message,
                    }
                )

                tokens += count_tokens(action_step.observation.message)

            # TODO: Truncate on self.max_tokens!

            assistant_message = {"role": "assistant"}

            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            if previous_node.assistant_message:
                assistant_message["content"] = previous_node.assistant_message
                tokens += count_tokens(previous_node.assistant_message)

            messages.append(assistant_message)
            messages.extend(tool_responses)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages

    @classmethod
    def create(cls, message_history_type: MessageHistoryType, **obj):
        obj["message_history_type"] = message_history_type

        if message_history_type == MessageHistoryType.REACT:
            from moatless.message_history.react import ReactMessageHistoryGenerator

            return ReactMessageHistoryGenerator(**obj)

        elif message_history_type == MessageHistoryType.MESSAGES_COMPACT:
            from moatless.message_history.compact import CompactMessageHistoryGenerator

            return CompactMessageHistoryGenerator(**obj)

        elif message_history_type == MessageHistoryType.SUMMARY:
            from moatless.message_history.summary import SummaryMessageHistoryGenerator

            return SummaryMessageHistoryGenerator(**obj)
        
        elif message_history_type == MessageHistoryType.MESSAGES:
            
            return cls(**obj)
        else:
            raise ValueError(f"Invalid message_history_type: {message_history_type}")

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "message_history_type" in obj:
            obj["message_history_type"] = MessageHistoryType(obj["message_history_type"])
            return cls.create(**obj)

        return obj
