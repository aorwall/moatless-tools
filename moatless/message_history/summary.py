import logging
from typing import List

from moatless.completion.schema import (
    ChatCompletionUserMessage,
    AllMessageValues,
)

from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node

logger = logging.getLogger(__name__)


class SummaryMessageHistoryGenerator(MessageHistoryGenerator):
    def generate_messages(self, node: Node) -> List[AllMessageValues]:
        previous_nodes = node.get_trajectory()

        formatted_history: List[str] = []
        counter = 0

        content = node.get_root().message
        if not previous_nodes:
            return [ChatCompletionUserMessage(role="user", content=content)]

        for i, previous_node in enumerate(previous_nodes):
            if previous_node.action:
                counter += 1
                formatted_state = f"\n\n## Step {counter}\n"
                if previous_node.action.thoughts:
                    formatted_state += f"Thoughts: {previous_node.action.thoughts}\n"
                formatted_state += f"Action: {previous_node.action.name}\n"
                formatted_state += previous_node.action.to_prompt()

                if previous_node.observation:
                    if (
                        hasattr(previous_node.observation, "summary")
                        and previous_node.observation.summary
                        and i < len(previous_nodes) - 1
                    ):
                        formatted_state += f"\n\nObservation: {previous_node.observation.summary}"
                    else:
                        formatted_state += f"\n\nObservation: {previous_node.observation.message}"
                else:
                    logger.warning(f"No output found for Node{previous_node.node_id}")
                    formatted_state += "\n\nObservation: No output found."

                formatted_history.append(formatted_state)

        # content += "\n\nBelow is the history of previously executed actions and their observations.\n"
        content += "<history>\n"
        content += "\n".join(formatted_history)
        content += "\n</history>\n\n"

        if self.include_file_context:
            content += "\n\nThe following code has already been viewed:\n"
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

        return [ChatCompletionUserMessage(role="user", content=content)]
