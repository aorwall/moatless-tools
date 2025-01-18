import logging
from typing import List

from pydantic import Field

from moatless.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
    AllMessageValues,
)
from moatless.message_history.compact import CompactMessageHistoryGenerator
from moatless.node import Node
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ReactMessageHistoryGenerator(CompactMessageHistoryGenerator):
    disable_thoughts: bool = Field(default=False, description="Do not include thoughts messages in the history")

    def generate_messages(self, node: Node) -> List[AllMessageValues]:
        node_messages = self.get_node_messages(node)
        logger.info(f"Node messages: {len(node_messages)}")

        messages = []

        # Convert node messages to react format
        for node_message in node_messages:
            if node_message.user_message:
                messages.append(ChatCompletionUserMessage(role="user", content=node_message.user_message))

            if node_message.assistant_message:
                messages.append(
                    ChatCompletionAssistantMessage(role="assistant", content=node_message.assistant_message)
                )

            if node_message.action:
                # Add thought and action message
                if self.disable_thoughts:
                    thought = ""
                else:
                    thought = (
                        f"Thought: {node_message.action.thoughts}" if hasattr(node_message.action, "thoughts") else ""
                    )

                action_str = f"Action: {node_message.action.name}"
                action_input = node_message.action.format_args_for_llm()

                if thought:
                    assistant_content = f"{thought}\n{action_str}"
                else:
                    assistant_content = action_str

                if action_input:
                    assistant_content += f"\n{action_input}"

                messages.append(ChatCompletionAssistantMessage(role="assistant", content=assistant_content))
                if node_message.observation:
                    messages.append(
                        ChatCompletionUserMessage(role="user", content=f"Observation: {node_message.observation}")
                    )

        tokens = count_tokens("".join([m["content"] for m in messages if m.get("content")]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages
