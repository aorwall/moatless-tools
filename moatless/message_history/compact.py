import logging
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.schema import ActionArguments
from moatless.actions.view_code import CodeSpan, ViewCodeArgs
from moatless.actions.view_diff import ViewDiffArgs
from moatless.completion.schema import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
)
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.node import Node
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class NodeMessage(BaseModel):
    user_message: Optional[str] = Field(default=None, description="The user message")
    assistant_message: Optional[str] = Field(default=None, description="The assistant message")
    action: Optional[ActionArguments] = Field(default=None, description="The action")
    observation: Optional[str] = Field(default=None, description="The observation")


class CompactMessageHistoryGenerator(MessageHistoryGenerator):
    message_cache: bool = Field(default=False, description="Cache the message history if the LLM supports it")

    async def generate_messages(self, node: Node) -> list[AllMessageValues]:
        messages = []

        node_messages = await self.get_node_messages(node)

        tool_idx = 0
        tokens = 0

        for action, observation in node_messages:
            tool_calls = []
            tool_idx += 1
            tool_call_id = f"tool_{tool_idx}"

            exclude = None
            content = None

            if not self.thoughts_in_action:
                exclude = {"thoughts"}
                content = action.thoughts

            tool_call = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": action.name,
                    "arguments": action.model_dump_json(exclude=exclude),
                },
            }

            tool_calls.append(tool_call)

            tokens += count_tokens(action.model_dump_json(exclude=exclude))

            messages.append(ChatCompletionAssistantMessage(role="assistant", tool_calls=tool_calls, content=content))
            messages.append(
                ChatCompletionToolMessage(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=f"Observation: {observation}",
                )
            )

            tokens += count_tokens(observation)

        logger.info(f"Generated {len(messages)} messages with {tokens} tokens")

        return messages

    async def get_node_messages(self, node: "Node") -> list[NodeMessage]:
        """
        Creates a list of (action, observation) tuples from the node's trajectory.
        Respects token limits while preserving ViewCode context.

        Returns:
            List of tuples where each tuple contains:
            - ActionArguments object
            - Observation summary string
        """
        previous_nodes = node.get_trajectory()
        logger.info(f"Previous nodes: {len(previous_nodes)}")
        if not previous_nodes:
            return []

        # Calculate initial token count
        total_tokens = node.file_context.context_size()
        total_tokens += count_tokens(node.get_root().message)

        # Pre-calculate test output tokens if there's a patch
        test_output_tokens = 0
        test_output = None
        run_tests_args = None
        if node.file_context.has_runtime and node.file_context.has_patch():
            if node.file_context.has_test_patch():
                thoughts = "Run the updated tests to verify the changes."
            else:
                thoughts = "Before adding new test cases I run the existing tests to verify regressions."

            run_tests_args = RunTestsArgs(thoughts=thoughts, test_files=list(node.file_context._test_files.keys()))

            test_output = ""

            # Add failure details if any
            failure_details = node.file_context.get_test_failure_details()
            if failure_details:
                test_output += failure_details + "\n\n"

            test_output += node.file_context.get_test_summary()
            test_output_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
            total_tokens += test_output_tokens

        node_messages = []
        shown_files = set()
        shown_diff = False  # Track if we've shown the first diff
        last_test_status = None  # Track the last test status

        for i, previous_node in enumerate(reversed(previous_nodes)):
            current_messages: list[NodeMessage] = []

            user_message = previous_node.user_message
            assistant_message = previous_node.assistant_message

            if previous_node.feedback_data and not user_message:
                user_message = previous_node.feedback_data.feedback

            if user_message or assistant_message:
                current_messages.append(NodeMessage(user_message=user_message, assistant_message=assistant_message))

            if previous_node.action_steps:
                for i, action_step in enumerate(previous_node.action_steps):
                    if action_step.action.name == "ViewCode":
                        # Always include ViewCode actions
                        file_path = action_step.action.files[0].file_path

                        if file_path not in shown_files:
                            context_file = previous_node.file_context.get_context_file(file_path)
                            if context_file and (context_file.span_ids or context_file.show_all_spans):
                                shown_files.add(context_file.file_path)
                                observation = context_file.to_prompt(
                                    show_span_ids=False,
                                    show_line_numbers=True,
                                    exclude_comments=False,
                                    show_outcommented_code=True,
                                    outcomment_code_comment="... rest of the code",
                                )
                            else:
                                observation = action_step.observation.message
                            current_messages.append(NodeMessage(action=action_step.action, observation=observation))
                    else:
                        # Count tokens for non-ViewCode actions
                        observation_str = (
                            action_step.observation.summary
                            if self.include_file_context
                            and hasattr(action_step.observation, "summary")
                            and action_step.observation.summary
                            else action_step.observation.message
                            if action_step.observation
                            else "No output found."
                        )

                        # Calculate tokens for this message pair
                        action_tokens = count_tokens(action_step.action.model_dump_json())
                        observation_tokens = count_tokens(observation_str)
                        message_tokens = action_tokens + observation_tokens

                        # Only add if within token limit
                        if total_tokens + message_tokens <= self.max_tokens:
                            total_tokens += message_tokens
                            current_messages.append(NodeMessage(action=action_step.action, observation=observation_str))
                        else:
                            # Skip remaining non-ViewCode messages if we're over the limit
                            continue

                # Handle file context for non-ViewCode actions
                if self.include_file_context and previous_node.action.name != "ViewCode":
                    files_to_show = set()
                    has_edits = False
                    context_files = previous_node.file_context.get_context_files()
                    for context_file in context_files:
                        if (
                            context_file.was_edited or context_file.was_viewed
                        ) and context_file.file_path not in shown_files:
                            files_to_show.add(context_file.file_path)
                        if context_file.was_edited:
                            has_edits = True

                    shown_files.update(files_to_show)

                    if files_to_show:
                        # Batch all files into a single ViewCode action
                        code_spans = []
                        observations = []

                        for file_path in files_to_show:
                            context_file = previous_node.file_context.get_context_file(file_path)
                            if context_file.show_all_spans:
                                code_spans.append(CodeSpan(file_path=file_path))
                            elif context_file.span_ids:
                                code_spans.append(
                                    CodeSpan(
                                        file_path=file_path,
                                        span_ids=context_file.span_ids,
                                    )
                                )
                            else:
                                continue

                            prompt = context_file.to_prompt(
                                show_span_ids=False,
                                show_line_numbers=True,
                                exclude_comments=False,
                                show_outcommented_code=True,
                                outcomment_code_comment="... rest of the code",
                            )
                            observations.append(prompt)

                        if code_spans:
                            thought = "Let's view the content in the updated files"
                            args = ViewCodeArgs(files=code_spans, thoughts=thought)
                            current_messages.append(NodeMessage(action=args, observation="\n\n".join(observations)))

                    # Show ViewDiff on first edit
                    if has_edits and self.include_git_patch and not shown_diff:
                        patch = node.file_context.generate_git_patch()
                        if patch:
                            view_diff_args = ViewDiffArgs(
                                thoughts="Let's review the changes made to ensure we've properly implemented everything required for the task. I'll check the git diff to verify the modifications."
                            )
                            diff_tokens = count_tokens(patch) + count_tokens(view_diff_args.model_dump_json())
                            if total_tokens + diff_tokens <= self.max_tokens:
                                total_tokens += diff_tokens
                                current_messages.append(
                                    NodeMessage(
                                        action=view_diff_args,
                                        observation=f"Current changes in workspace:\n```diff\n{patch}\n```",
                                    )
                                )
                                shown_diff = True

                    # Add test results only if status changed or first occurrence
                    if (
                        node.file_context.has_runtime
                        and previous_node.observation
                        and previous_node.observation.properties.get("diff")
                    ):
                        current_test_status = node.file_context.get_test_status()
                        if last_test_status is None or current_test_status != last_test_status:
                            if node.file_context.has_test_patch():
                                thoughts = "Run the updated tests to verify the changes."
                            else:
                                thoughts = (
                                    "Before adding new test cases I run the existing tests to verify regressions."
                                )

                            run_tests_args = RunTestsArgs(
                                thoughts=thoughts,
                                test_files=list(node.file_context._test_files.keys()),
                            )

                            test_output = ""
                            if last_test_status is None:
                                # Show full details for first test run
                                failure_details = node.file_context.get_test_failure_details()
                                if failure_details:
                                    test_output += failure_details + "\n\n"

                            test_output += node.file_context.get_test_summary()

                            # Calculate and check token limits
                            test_tokens = count_tokens(test_output) + count_tokens(run_tests_args.model_dump_json())
                            if total_tokens + test_tokens <= self.max_tokens:
                                total_tokens += test_tokens
                                current_messages.append(NodeMessage(action=run_tests_args, observation=test_output))
                                last_test_status = current_test_status

            node_messages = current_messages + node_messages

        logger.info(f"Generated message history with {total_tokens} tokens")
        return node_messages
