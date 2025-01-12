import logging
from typing import List, Any

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)
from pydantic import BaseModel, Field, field_serializer

from moatless.actions.model import ActionArguments
from moatless.actions.run_tests import RunTestsArgs
from moatless.actions.view_code import ViewCodeArgs, CodeSpan
from moatless.actions.view_diff import ViewDiffArgs
from moatless.node import Node
from moatless.schema import MessageHistoryType
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class MessageHistoryGenerator(BaseModel):
    message_history_type: MessageHistoryType = Field(
        default=MessageHistoryType.MESSAGES,
        description="Type of message history to generate",
    )
    include_file_context: bool = Field(
        default=True, description="Whether to include file context in messages"
    )
    include_git_patch: bool = Field(
        default=True, description="Whether to include git patch in messages"
    )
    include_root_node: bool = Field(default=True)
    max_tokens: int = Field(
        default=20000, description="Maximum number of tokens allowed in message history"
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )
    enable_index_in_tool_call: bool = Field(
        default=True, description="Whether to include index in the tool call"
    )

    model_config = {
        "ser_json_timedelta": "iso8601",
        "ser_json_bytes": "base64",
        "ser_json_inf_nan": "null",
        "json_schema_serialization_defaults": True,
        "json_encoders": None,  # Remove this as it's v1 syntax
    }

    def __init__(self, **data: Any):
        super().__init__(**data)

    @field_serializer("message_history_type")
    def serialize_message_history_type(
        self, message_history_type: MessageHistoryType
    ) -> str:
        return message_history_type.value

    def generate(self, node: "Node") -> List[AllMessageValues]:  # type: ignore
        logger.debug(
            f"Generating message history for Node{node.node_id}: {self.message_history_type}"
        )
        generators = {
            MessageHistoryType.SUMMARY: self._generate_summary_history,
            MessageHistoryType.REACT: self._generate_react_history,
            MessageHistoryType.MESSAGES: self._generate_message_history,
            MessageHistoryType.MESSAGES_COMPACT: self._generate_compact_message_history,
        }
        start_idx = 0 if self.include_root_node else 1
        previous_nodes = node.get_trajectory()[start_idx:]
        return generators[self.message_history_type](node, previous_nodes)

    def _generate_message_history(
        self, node: Node, previous_nodes: List["Node"]
    ) -> List[dict[str, Any]]:
        messages = []
        tool_idx = 0
        tokens = 0

        for i, previous_node in enumerate(previous_nodes):
            # Handle user message
            if previous_node.user_message:
                message_content = [{"type": "text", "text": previous_node.user_message}]

                if previous_node.artifact_changes:
                    for change in previous_node.artifact_changes:
                        artifact = previous_node.workspace.get_artifact_by_id(
                            change.artifact_id
                        )
                        if artifact:
                            message = f"{artifact.type} artifact: {artifact.id}"
                            message_content.append({"type": "text", "text": message})
                            message_content.append(artifact.to_prompt_format())

                messages.append(
                    ChatCompletionUserMessage(role="user", content=message_content)
                )
                tokens += count_tokens(previous_node.user_message)

            tool_calls = []
            tool_responses = []

            if previous_node.feedback_data:
                messages.append(
                    ChatCompletionUserMessage(
                        role="user", content=previous_node.feedback_data.feedback
                    )
                )

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
                            "arguments": action_step.action.model_dump_json(
                                exclude=exclude
                            ),
                        },
                    }
                )

                tokens += count_tokens(
                    action_step.action.model_dump_json(exclude=exclude)
                )

                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": action_step.observation.message,
                    }
                )

                tokens += count_tokens(action_step.observation.message)

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

    def _generate_compact_message_history(
        self, node: Node, previous_nodes: List["Node"]
    ) -> List[dict[str, Any]]:
        messages = [
            ChatCompletionUserMessage(role="user", content=node.get_root().message)
        ]

        if len(previous_nodes) <= 1:
            return messages

        node_messages = self.get_node_messages(node)

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

            messages.append(
                ChatCompletionAssistantMessage(
                    role="assistant", tool_calls=tool_calls, content=content
                )
            )
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

    def _generate_react_history(
        self, node: "Node", previous_nodes: List["Node"]
    ) -> List[AllMessageValues]:
        messages = [
            ChatCompletionUserMessage(role="user", content=node.get_root().message)
        ]

        if len(previous_nodes) <= 1:
            return messages

        node_messages = self.get_node_messages(node)

        # Convert node messages to react format
        for action, observation in node_messages:
            # Add thought and action message
            thought = (
                f"Thought: {action.thoughts}" if hasattr(action, "thoughts") else ""
            )
            action_str = f"Action: {action.name}"
            action_input = action.format_args_for_llm()

            assistant_content = f"{thought}\n{action_str}"
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

    def get_node_messages(self, node: "Node") -> List[tuple[ActionArguments, str]]:
        """
        Creates a list of (action, observation) tuples from the node's trajectory.
        Respects token limits while preserving ViewCode context.

        Returns:
            List of tuples where each tuple contains:
            - ActionArguments object
            - Observation summary string
        """
        previous_nodes = node.get_trajectory()[:-1]
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
                thoughts = (
                    "<thoughts>Run the updated tests to verify the changes.</thoughts>"
                )
            else:
                thoughts = "<thoughts>Before adding new test cases I run the existing tests to verify regressions.</thoughts>"

            run_tests_args = RunTestsArgs(
                thoughts=thoughts, test_files=list(node.file_context._test_files.keys())
            )

            test_output = ""

            # Add failure details if any
            failure_details = node.file_context.get_test_failure_details()
            if failure_details:
                test_output += failure_details + "\n\n"

            test_output += node.file_context.get_test_summary()
            test_output_tokens = count_tokens(test_output) + count_tokens(
                run_tests_args.model_dump_json()
            )
            total_tokens += test_output_tokens

        node_messages = []
        shown_files = set()
        shown_diff = False  # Track if we've shown the first diff
        last_test_status = None  # Track the last test status

        for i, previous_node in enumerate(reversed(previous_nodes)):
            current_messages = []

            if previous_node.action_steps:
                for i, action_step in enumerate(previous_node.action_steps):
                    # FIXME: Make consistent way of handling thoughts!
                    if i == 0:
                        action_step.action.thoughts = previous_node.assistant_message

                    if action_step.action.name == "ViewCode":
                        # Always include ViewCode actions
                        file_path = action_step.action.files[0].file_path

                        if file_path not in shown_files:
                            context_file = previous_node.file_context.get_context_file(
                                file_path
                            )
                            if context_file and (
                                context_file.span_ids or context_file.show_all_spans
                            ):
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
                            current_messages.append((action_step.action, observation))
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
                        action_tokens = count_tokens(
                            action_step.action.model_dump_json()
                        )
                        observation_tokens = count_tokens(observation_str)
                        message_tokens = action_tokens + observation_tokens

                        # Only add if within token limit
                        if total_tokens + message_tokens <= self.max_tokens:
                            total_tokens += message_tokens
                            current_messages.append(
                                (action_step.action, observation_str)
                            )
                        else:
                            # Skip remaining non-ViewCode messages if we're over the limit
                            continue

                # Handle file context for non-ViewCode actions
                if (
                    self.include_file_context
                    and previous_node.action.name != "ViewCode"
                ):
                    files_to_show = set()
                    has_edits = False
                    for context_file in previous_node.file_context.get_context_files():
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
                            context_file = previous_node.file_context.get_context_file(
                                file_path
                            )
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

                            observations.append(
                                context_file.to_prompt(
                                    show_span_ids=False,
                                    show_line_numbers=True,
                                    exclude_comments=False,
                                    show_outcommented_code=True,
                                    outcomment_code_comment="... rest of the code",
                                )
                            )

                        if code_spans:
                            thought = f"<thoughts>Let's view the content in the updated files</thoughts>"
                            args = ViewCodeArgs(files=code_spans, thoughts=thought)
                            current_messages.append((args, "\n\n".join(observations)))

                    # Show ViewDiff on first edit
                    if has_edits and self.include_git_patch and not shown_diff:
                        patch = node.file_context.generate_git_patch()
                        if patch:
                            view_diff_args = ViewDiffArgs(
                                thoughts="<thoughts>Let's review the changes made to ensure we've properly implemented everything required for the task. I'll check the git diff to verify the modifications.</thoughts>"
                            )
                            diff_tokens = count_tokens(patch) + count_tokens(
                                view_diff_args.model_dump_json()
                            )
                            if total_tokens + diff_tokens <= self.max_tokens:
                                total_tokens += diff_tokens
                                current_messages.append(
                                    (
                                        view_diff_args,
                                        f"Current changes in workspace:\n```diff\n{patch}\n```",
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
                        if (
                            last_test_status is None
                            or current_test_status != last_test_status
                        ):
                            if node.file_context.has_test_patch():
                                thoughts = "<thoughts>Run the updated tests to verify the changes.</thoughts>"
                            else:
                                thoughts = "<thoughts>Before adding new test cases I run the existing tests to verify regressions.</thoughts>"

                            run_tests_args = RunTestsArgs(
                                thoughts=thoughts,
                                test_files=list(node.file_context._test_files.keys()),
                            )

                            test_output = ""
                            if last_test_status is None:
                                # Show full details for first test run
                                failure_details = (
                                    node.file_context.get_test_failure_details()
                                )
                                if failure_details:
                                    test_output += failure_details + "\n\n"

                            test_output += node.file_context.get_test_summary()

                            # Calculate and check token limits
                            test_tokens = count_tokens(test_output) + count_tokens(
                                run_tests_args.model_dump_json()
                            )
                            if total_tokens + test_tokens <= self.max_tokens:
                                total_tokens += test_tokens
                                current_messages.append((run_tests_args, test_output))
                                last_test_status = current_test_status

            # Add current messages to the beginning of the list
            node_messages = current_messages + node_messages

        logger.info(f"Generated message history with {total_tokens} tokens")
        return node_messages
