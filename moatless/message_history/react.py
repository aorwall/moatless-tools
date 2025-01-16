import logging
from typing import List

from litellm import Field

from moatless.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage, AllMessageValues, )
from moatless.message_history.compact import CompactMessageHistoryGenerator
from moatless.node import Node
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ReactMessageHistoryGenerator(CompactMessageHistoryGenerator):
    disable_thoughts: bool = Field(default=False, description="Do not include thoughts messages in the history")

    def generate_messages(
        self, node: Node
    ) -> List[AllMessageValues]:
        previous_nodes = node.get_trajectory()

        messages = [
            ChatCompletionUserMessage(role="user", content=node.get_root().message)
        ]

        if len(previous_nodes) <= 1:
            return messages

        node_messages = self.get_node_messages(node)

        # Convert node messages to react format
        for action, observation in node_messages:
            # Add thought and action message
            if self.disable_thoughts:
                thought = ""
            else:
                thought = (
                    f"Thought: {action.thoughts}" if hasattr(action, "thoughts") else ""
                )

                
            action_str = f"Action: {action.name}"
            action_input = action.format_args_for_llm()

            if thought:
                assistant_content = f"{thought}\n{action_str}"
            else:
                assistant_content = action_str

            if action_input:
                assistant_content += f"\n{action_input}"

            messages.append(
                ChatCompletionAssistantMessage(
                    role="assistant", content=assistant_content
                )
            )
            messages.append(
                ChatCompletionUserMessage(
                    role="user", content=f"Observation: {observation}"
                )
            )

        tokens = count_tokens(
            "".join([m["content"] for m in messages if m.get("content")])
        )
        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")
        return messages

    def _generate_summary_history(
        self, node: Node, previous_nodes: List[Node]
    ) -> List[AllMessageValues]:
        formatted_history: List[str] = []
        counter = 0

        if self.include_root_node:
            content = node.get_root().message
            if not previous_nodes:
                return [ChatCompletionUserMessage(role="user", content=content)]
        else:
            content = ""
            if not previous_nodes:
                return []

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = f"\n\n## Step {counter}\n"
                if not self.disable_thoughts and previous_node.action.thoughts:
                    formatted_state += f"Thoughts: {previous_node.action.thoughts}\n"
                formatted_state += f"Action: {previous_node.action.name}\n"
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    if (
                        hasattr(previous_node.observation, "summary")
                        and previous_node.observation.summary
                        and i < len(previous_nodes) - 1
                    ):
                        formatted_state += (
                            f"\n\nObservation: {previous_node.observation.summary}"
                        )
                    else:
                        formatted_state += (
                            f"\n\nObservation: {previous_node.observation.message}"
                        )
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")
                    formatted_state += "\n\nObservation: No output found."

                formatted_history.append(formatted_state)

        content += "\n\nBelow is the history of previously executed actions and their observations.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"

        if self.include_file_context:
            content += "\n\nHere's the code that is currently in context:\n"
            content += node.file_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )

        if self.include_git_patch:
            git_patch = node.file_context.generate_git_patch()
            if git_patch:
                content += "\n\nThe current git diff is:\n"
                content += "```diff\n"
                content += git_patch
                content += "\n```"

        current_test_status = node.file_context.get_test_status()
        
        if node.file_context.has_patch() and node.file_context.test_files:

            response_msg = ""
            if node.file_context.has_test_patch():
                response_msg = "The following tests have been run:\n"
            else:
                response_msg = "The following existing tests have been run to verify regressions in."

            if node.file_context.test_files:
                for test_file in node.file_context.test_files:
                    response_msg += f"* {test_file.file_path}\n"

            failure_details = node.file_context.get_test_failure_details()
            if failure_details:
                response_msg += f"\n{failure_details}"
            
            summary = f"\n{node.file_context.get_test_summary()}"
            response_msg += summary

        return [ChatCompletionUserMessage(role="user", content=content)]
