import logging
from typing import List

from pydantic import Field

from moatless.completion.schema import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from moatless.message_history.compact import CompactMessageHistoryGenerator
from moatless.node import Node
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ReactCompactMessageHistoryGenerator(CompactMessageHistoryGenerator):
    disable_thoughts: bool = Field(default=False, description="Do not include thoughts messages in the history")

    async def generate_messages(self, node: Node) -> list[AllMessageValues]:
        node_messages = await self.get_node_messages(node)
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

            if not node_message.actions:
                continue

            assistant_content = ""
            user_content = "Observations:\n "

            has_actions = False

            for action, observation in zip(node_message.actions, node_message.observations, strict=True):
                from moatless.actions.think import ThinkArgs

                if isinstance(action, ThinkArgs):
                    assistant_content += f"Thought: {action.thought}\n"
                    continue
                elif hasattr(action, "thoughts") and action.thoughts:
                    assistant_content += f"Thought: {action.thoughts}\n"
                else:
                    assistant_content += "\n"

                has_actions = True

                assistant_content += f"Action: {action.name}\n"
                assistant_content += action.format_args_for_llm()

                if observation:
                    user_content += f"{observation}\n"

            if has_actions:
                messages.append(ChatCompletionAssistantMessage(role="assistant", content=assistant_content))
                messages.append(ChatCompletionUserMessage(role="user", content=user_content))

        tokens = count_tokens("".join([m["content"] for m in messages if m.get("content")]))
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages
