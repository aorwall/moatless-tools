from enum import Enum
import json
import logging
from typing import Optional, List, Any

from pydantic import ConfigDict, Field, PrivateAttr, BaseModel, model_validator

from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup
from moatless.edit.clarify import _get_post_end_line_index, _get_pre_start_line
from moatless.edit.prompt import PLAN_TO_CODE_SYSTEM_PROMPT
from moatless.index.code_index import is_test
from moatless.repository import CodeFile
from moatless.schema import VerificationIssueType, FileWithSpans, ChangeType
from moatless.state import AgenticState, ActionRequest, StateOutcome, AssistantMessage, Message, UserMessage, TakeAction
from moatless.utils.tokenizer import count_tokens

from moatless.verify.lint import VerificationIssue

logger = logging.getLogger("PlanToCode")

class RequestCodeChange(ActionRequest):
    """
    Request for the next code change.
    """

    scratch_pad: str = Field(..., description="Your step by step reasoning on how to do the code change and whats the next step is.")

    change_type: ChangeType = Field(
        ..., description="A string that can be set to 'addition', 'modification', or 'deletion'. 'Addition' refers to adding a new function or class, 'modification' refers to changing existing code, and 'deletion' refers to removing a function or class."
    )
    instructions: str = Field(
        ..., description="Instructions about the next step to do the code change."
    )
    start_line: int = Field(
        ..., description="The start line of the existing code to be updated."
    )
    end_line: int = Field(..., description="The end line of the code to be updated when modifying existing code.")

    pseudo_code: str = Field(
        ..., description="Pseudo code for the code change."
    )
    file_path: str = Field(
        ..., description="The file path of the code to be updated."
    )

    planned_steps: List[str] = Field(
        default_factory=list,
        description="Planned steps that should be executed after the current step."
    )

    #@model_validator(mode="before")
    #@classmethod
    def validate_steps(cls, data: Any):
        # The Antrophic API sometimes returns steps as a string instead of a list
        logger.info(f"validate_steps {isinstance(data, dict) and 'steps' in data} {isinstance(data['steps'], str)}")
        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], str):
            logger.info(f"validate_steps: Converting steps to list: {data['steps']}")
            data["steps"] = json.loads(data["steps"])
            logger.info(f"validate_steps: Converted steps to list: {data['steps']}")
        return data



class RequestMoreContext(ActionRequest):
    """
    Request to see code that is not in the current context.
    """
    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    files: List[FileWithSpans] = Field(
        ..., description="The code that should be provided in the file context."
    )


class Review(ActionRequest):

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    instructions: str = Field(
        ..., description="Review instructions."
    )

class Finish(ActionRequest):

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    reason: str = Field(
        ..., description="Finish the request and explain why"
    )

class Reject(ActionRequest):

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    reason: str = Field(
        ..., description="Reject the request and explain why."
    )


class PlanRequest(TakeAction):
    """
    Request to apply a change to the code.
    """

    @classmethod
    def available_actions(cls) -> List[ActionRequest]:
        return [
            RequestCodeChange,
            RequestMoreContext,
            Review,
            Finish,
            Reject,
        ]

class PlanToCode(AgenticState):
    message: Optional[str] = Field(
        None,
        description="Message from last transitioned state",
    )

    diff: Optional[str] = Field(
        None,
        description="The diff of a previous code change.",
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    max_tokens_in_edit_prompt: int = Field(
        1000,
        description="The maximum number of tokens in a span to show in the edit prompt.",
    )

    min_tokens_in_edit_prompt: int = Field(
        50,
        description="The minimum number of tokens in a span to show in the edit prompt.",
    )

    finish_on_review: bool = Field(
        False, description="Whether to finish the task if a review is requested."
    )

    max_repeated_test_failures: int = Field(
        3,
        description="The maximum number of repeated test failures before rejecting the task.",
    )

    max_repated_git_diffs: int = Field(
        2,
        description="The maximum number of repeated git diffs before rejecting the task.",
    )

    include_message_history: bool = Field(
        True,
        description="Whether to include the message history in the prompt.",
    )

    verify: bool = Field(True, description="Whether to run verification job before executing the next action.")

    verification_issues: list[VerificationIssue] | None = Field(
        None,
        description="Verification errors.",
    )

    _expanded_context: bool = PrivateAttr(False)

    def _execute_action(self, action: TakeAction) -> StateOutcome:
        if isinstance(action.action, Review):
            if self.diff and self.finish_on_review:
                logger.info("Review suggested after diff, will finish")
                return StateOutcome.transition(
                    trigger="finish", output={"message": "Finish on suggested review."}
                )
            else:
                return StateOutcome.retry(
                    "Review isn't possible. If the change is done you can finish or reject the task."
                )

        if isinstance(action.action, Finish):
            return StateOutcome.transition(
                trigger="finish", output={"message": action.action.reason}
            )
        elif isinstance(action.action, Reject):
            return StateOutcome.transition(
                trigger="reject", output={"message": action.action.reason}
            )

        elif isinstance(action.action, RequestCodeChange):
            return self._request_for_change(action.action)

        elif isinstance(action.action, RequestMoreContext):
            return self._request_more_context(action.action)
        return StateOutcome.retry(
            "You must either provide an apply_change action or finish."
        )

    def init(self) -> Optional[StateOutcome]:
        if not self.file_context.files:
            raise ValueError("No files in the file context.")

        previous_states = self.get_previous_states(self)

        diff_counts = {}
        verification_issue_counts = {}
        for state in previous_states:
            if state.diff:
                diff_counts[state.diff] = diff_counts.get(state.diff, 0) + 1
            if state.verification_issues:
                for issue in state.verification_issues:
                    issue_key = f"{issue.file_path}:{issue.span_id}:{issue.message}"
                    verification_issue_counts[issue_key] = verification_issue_counts.get(issue_key, 0) + 1

        # Check if any diff exceeds the maximum allowed repetitions
        for diff, count in diff_counts.items():
            if count > self.max_repated_git_diffs:
                return StateOutcome.reject(f"The following diff has been repeated {count} times, which exceeds the maximum allowed repetitions of {self.max_repated_git_diffs}:\n\n{diff}")

        # Check if any verification issue exceeds the maximum allowed repetitions
        for issue_key, count in verification_issue_counts.items():
            if count > self.max_repeated_test_failures:
                return StateOutcome.reject(f"The following verification issue has been repeated {count} times, which exceeds the maximum allowed repetitions of {self.max_repeated_test_failures}:\n\n{issue_key}")

        if self.verify:
            # Run all test files that are in context
            test_files = [file for file in self.file_context.files if is_test(file.file_path)]
            if not test_files:
                logger.info(f"{self.name}:{self.id} No test files in the file context, will not run tests.")
            else:
                test_file_paths = [file.file_path for file in test_files]
                self.verification_issues = self.workspace.run_tests(test_file_paths)
                if self.verification_issues:
                    logger.info(
                        f"{self.name}:{self.id} Tests returned {len(self.verification_issues)} verification issues."
                    )

                    all_transitions_issues = [self.verification_issues]
                    all_transitions_issues.extend([state.verification_issues for state in previous_states if state.verification_issues])

                    for issues in all_transitions_issues:
                        issue_keys = []
                        for issue in issues:
                            issue_keys.append(f"{issue.file_path}:{issue.span_id}:{issue.message}")

                        issue_key = "\n".join(issue_keys)
                        verification_issue_counts[issue_key] = verification_issue_counts.get(issue_key, 0) + 1

                    # Check if any verification issue exceeds the maximum allowed repetitions
                    for issue_key, count in verification_issue_counts.items():
                        if count > self.max_repeated_test_failures:
                            return StateOutcome.reject(
                                f"The following verification issue has been repeated {count} times, which exceeds the maximum allowed repetitions of {self.max_repeated_test_failures}:\n\n{issue_key}")

                    # Keep file context size down by replacing spans with failing spans
                    failed_test_spans_by_file_path: dict = {}
                    for issue in self.verification_issues:
                        failed_test_spans_by_file_path.setdefault(issue.file_path, []).append(issue.span_id)

                    for test_file in test_files:
                        # TODO: Find a way to rank spans to keep the most relevant ones
                        test_file.remove_all_spans()
                        failed_span_ids = failed_test_spans_by_file_path.get(test_file.file_path)
                        if failed_span_ids:
                            test_file.add_spans(failed_span_ids)

    def action_type(self) -> type[PlanRequest]:
        return PlanRequest

    def _request_more_context(self, action: RequestMoreContext) -> StateOutcome:
        logger.info(f"{self.name}:{self.id}:RequestMoreContext: {action.files}")

        retry_message = ""
        for file_with_spans in action.files:
            file = self.file_repo.get_file(file_with_spans.file_path)
            if not file:
                logger.info(f"{self.name}:{self.id}:RequestMoreContext: {file_with_spans.file_path} is not found in the file repository.")
                return StateOutcome.retry(
                    f"File {file.file_path} is not found in the file repository."
                )

            missing_span_ids = set()
            suggested_span_ids = set()
            found_span_ids = set()
            for span_id in file_with_spans.span_ids:
                block_span = file.module.find_span_by_id(span_id)
                if not block_span:
                    # Try to find the relevant code block by code block identifier
                    block_identifier = span_id.split(".")[-1]
                    blocks = file.module.find_blocks_with_identifier(block_identifier)

                    if not blocks:
                        missing_span_ids.add(span_id)
                    elif len(blocks) > 1:
                        for block in blocks:
                            if block.belongs_to_span.span_id not in suggested_span_ids:
                                suggested_span_ids.add(block.belongs_to_span.span_id)
                    else:
                        block_span = blocks[0].belongs_to_span

                if block_span:
                    if block_span.initiating_block.type == CodeBlockType.CLASS:
                        if block_span.initiating_block.sum_tokens() < self.max_tokens_in_edit_prompt:
                            found_span_ids.add(block_span.span_id)
                            for child_span_id in block_span.initiating_block.span_ids:
                                found_span_ids.add(child_span_id)
                        else:
                            retry_message += f"Class {block_span.initiating_block.identifier} has too many tokens. Specify which functions to include..\n"
                            suggested_span_ids.update(block_span.initiating_block.span_ids)
                    else:
                        found_span_ids.add(block_span.span_id)

            if missing_span_ids:
                logger.info(f"{self.name}:{self.id}:RequestMoreContext: Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}")
                retry_message += f"Spans not found in {file_with_spans.file_path}: {', '.join(missing_span_ids)}\n"
            else:
                self.file_context.add_spans_to_context(file.file_path, found_span_ids)

            if retry_message and suggested_span_ids:
                logger.info(f"{self.name}:{self.id}:RequestMoreContext: Suggested spans: {', '.join(suggested_span_ids)}")
                retry_message += f"Did you mean one of these spans: {', '.join(suggested_span_ids)}\n"

        if retry_message:
            return StateOutcome.retry(retry_message)

        self.file_context.add_files_with_spans(action.files)

        message = "Added new spans:\n"
        for file in action.files:
            message += f" * {file.file_path} ({', '.join(file.span_ids)})\n"

        return StateOutcome.stay_in_state(output={"message": message})

    def _request_for_change(self, rfc: RequestCodeChange) -> StateOutcome:
        logger.info(
            f"{self.name}:{self.id}:RequestCodeChange: file_path={rfc.file_path}, start_line={rfc.start_line}, end_line={rfc.end_line}, change_type={rfc.change_type}"
        )

        if not rfc.instructions:
            return StateOutcome.retry(
                f"Please provide instructions for the code change."
            )

        if not rfc.start_line:
            return StateOutcome.retry(
                f"Please provide the start line for the code change."
            )

        if not rfc.end_line:
            if rfc.change_type != ChangeType.addition:
                return StateOutcome.retry(
                    f"If the intention is to modify an existing code span you must provide the end line for the code change."
                )

            logger.info(f"{self.name}:{self.id}:RequestCodeChange: End line not set, set to start line {rfc.start_line}")
            rfc.end_line = rfc.start_line

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            logger.warning(
                f"{self.name}:{self.id}:RequestCodeChange:  File {rfc.file_path} is not found in the file context."
            )

            files_str = ""
            for file in self.file_context.files:
                files_str += f" * {file.file_path}\n"

            return StateOutcome.retry(
                f"File {rfc.file_path} is not found in the file context. "
                f"You can only request changes to files that are in file context:\n{files_str}"
            )

        code_lines = context_file.file.content.split("\n")
        lines_to_edit = code_lines[
            rfc.start_line - 1: rfc.end_line
        ]
        code_to_edit = "\n".join(lines_to_edit)

        tokens = count_tokens(code_to_edit)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = (f"Lines {rfc.start_line} - {rfc.end_line} has {tokens} tokens, which is higher than the "
                           f"maximum allowed {self.max_tokens_in_edit_prompt} tokens.")
            logger.info(f"{self.name}:{self.id} {clarify_msg}. Ask for clarification.")
            return StateOutcome.retry(
                f"{clarify_msg}. Narrow down the instructions and specify the exact part of the code that needs to be "
                f"updated to fulfill the change. ")

        start_line, end_line, change_type = self.get_line_span(
            rfc.change_type, context_file.file, rfc.start_line, rfc.end_line, self.max_tokens_in_edit_prompt
        )

        logger.info(
            f"{self.name}:{self.id} Requesting code change in {rfc.file_path} from {start_line} to {end_line}"
        )

        span_ids = []
        span_to_update = context_file.file.module.find_spans_by_line_numbers(start_line, end_line)
        if span_to_update:
            # Pin the spans that are planned to be updated to context
            span_ids = [span.span_id for span in span_to_update]
            self.file_context.add_spans_to_context(rfc.file_path, span_ids=set(span_ids), pinned=True)

        # Add the two most relevant test files to file context if there are none to trigger tests on next iteration
        has_test_files = any(file for file in self.file_context.files if is_test(file.file_path))
        if not has_test_files:
            test_files_with_spans = self.workspace.code_index.find_test_files(rfc.file_path, query=code_to_edit, max_results=2, max_spans=1)
            self.file_context.add_files_with_spans(test_files_with_spans)

        return StateOutcome.transition(
            trigger="edit_code",
            output={
                "instructions": rfc.instructions,
                "pseudo_code": rfc.pseudo_code,
                "file_path": rfc.file_path,
                "change_type": change_type.value,
                "start_line": start_line,
                "end_line": end_line,
                "span_ids": span_ids
            }
        )

    def get_line_span(
        self,
        change_type: ChangeType,
        file: CodeFile,
        start_line: int,
        end_line: int,
        max_tokens: int,
    ) -> tuple[Optional[int], Optional[int], Optional[ChangeType]]:
        start_block = file.module.find_first_by_start_line(start_line)

        # Set just one line on additions which doesn't point to a specific code block
        if not start_block:
            if change_type == ChangeType.addition and end_line != start_line:
                logger.info(f"{self.name}:{self.id} Change type is addition, set end line to start line {start_line}, endline was {end_line}")
                return start_line, start_line, change_type.addition
            elif start_line == end_line:
                logger.info(f"{self.name}:{self.id} Start line {start_line} is equal to end line, expect addition ")
                return start_line, start_line, change_type.addition

        if (start_block and start_block.start_line == start_line
            and start_block.type.group == CodeBlockTypeGroup.STRUCTURE
            and not self.file_context.has_span(file.file_path, start_block.belongs_to_span.span_id)
            and change_type == ChangeType.addition
            and not file.module.find_first_by_start_line(start_line - 1)):
            logger.info(f"{self.name}:{self.id} Start block {start_block.display_name} at line {start_line} isn't "
                        f"in context, expect this to be an addition on line {start_line - 1} before the block")
            return start_line - 1, start_line - 1, change_type.addition

        if not start_block:
            structure_block = file.module
            logger.info(f"{self.name}:{self.id} Start block not found, set module as structure block")
        elif start_block.type.group == CodeBlockTypeGroup.STRUCTURE and (
            not end_line or start_block.end_line >= end_line
        ):
            structure_block = start_block
            logger.info(f"{self.name}:{self.id} Start block {start_block.display_name} is a structure block")
        else:
            structure_block = start_block.find_type_group_in_parents(
                CodeBlockTypeGroup.STRUCTURE
            )
            logger.info(f"{self.name}:{self.id} Set parent {structure_block.display_name} as structure block")

        structure_block_tokens = structure_block.sum_tokens()
        if structure_block_tokens > self.min_tokens_in_edit_prompt and structure_block_tokens < max_tokens:
            logger.info(
                f"{self.name}:{self.id} Return start and endline for block {structure_block.display_name} "
                f"{structure_block.start_line} - {structure_block.end_line} ({self.min_tokens_in_edit_prompt} "
                f"(min tokens) < {structure_block_tokens} (block tokens) < {max_tokens} (max tokens))"
            )
            return structure_block.start_line, structure_block.end_line, change_type

        if structure_block_tokens < max_tokens:
            previous_block = structure_block.find_last_previous_block_with_block_group(
                CodeBlockTypeGroup.STRUCTURE
            )
            if (
                previous_block
                and structure_block_tokens + previous_block.sum_tokens() < max_tokens
            ):
                start_line = previous_block.start_line
                structure_block_tokens += previous_block.sum_tokens()
                logger.info(
                    f"{self.name}:{self.id} Start from start line of the previous block {previous_block.display_name} that fits in the prompt"
                )
            else:
                start_line = structure_block.start_line

            next_structure_block = structure_block.find_next_block_with_block_group(
                CodeBlockTypeGroup.STRUCTURE
            )
            if (
                next_structure_block
                and structure_block_tokens + next_structure_block.sum_tokens()
                < max_tokens
            ):
                end_line = next_structure_block.end_line
                structure_block_tokens += next_structure_block.sum_tokens()
                logger.info(
                    f"{self.name}:{self.id} End at end line of the next block {next_structure_block.display_name} that fits in the prompt, at line {end_line}"
                )
            else:
                end_line = structure_block.end_line

            logger.info(
                f"{self.name}:{self.id} Return block [{structure_block.display_name}] ({start_line} - {end_line}) with {structure_block_tokens} tokens that covers the specified line span ({start_line} - {end_line})"
            )
            return start_line, end_line, change_type.modification

        if not end_line:
            end_line = start_line

        original_lines = file.content.split("\n")
        if structure_block.end_line - end_line < 5:
            logger.info(
                f"{self.name}:{self.id} Set structure block [{structure_block.display_name}] end line {structure_block.end_line} as it's {structure_block.end_line - end_line} lines from the end of the file"
            )
            end_line = structure_block.end_line
        else:
            end_line = _get_post_end_line_index(
                end_line, structure_block.end_line, original_lines
            )
            logger.info(f"{self.name}:{self.id} Set end line to {end_line}, structure block {structure_block.display_name} ends at line {structure_block.end_line}")

        if start_line - structure_block.start_line < 5:
            logger.info(
                f"{self.name}:{self.id} Set structure block [{structure_block.display_name}] start line {structure_block.start_line} as it's {start_line - structure_block.start_line} lines from the start of the file"
            )
            start_line = structure_block.start_line
        else:
            start_line = _get_pre_start_line(
                start_line, structure_block.start_line, original_lines
            )
            logger.info(
                f"{self.name}:{self.id} Set start line to {start_line}, structure block {structure_block.display_name} starts at line {structure_block.start_line}"
            )

        return start_line, end_line, change_type.modification

    def system_prompt(self) -> str:
        return PLAN_TO_CODE_SYSTEM_PROMPT

    def to_message(self, verbose: bool = True) -> str:
        response_msg = ""

        if self.message:
            response_msg += self.message

        if self.diff:
            response_msg += f"\n\n<diff>\n{self.diff}\n</diff>"

        if self.verification_issues:
            response_msg += "\n\nVerification issues:"

            for issue in self.verification_issues:
                if issue.type in [VerificationIssueType.RUNTIME_ERROR, VerificationIssueType.SYNTAX_ERROR]:
                    if verbose:
                        response_msg += f"\n\n<error>\n{issue.message}\n</error>"
                    else:
                        last_line = issue.message.split("\n")[-1]
                        response_msg += f"\n\n{last_line}"

                elif issue.type == VerificationIssueType.TEST_FAILURE:
                    response_msg += f"\n\nThe test {issue.span_id} in {issue.file_path} failed."

                    if verbose:
                        response_msg += f"Output:\n``` \n{issue.message}\n```\n\n"

        return response_msg

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        if self.initial_message:
            content = f"<issue>\n{self.initial_message}\n</issue>\n"
        else:
            content = ""

        previous_states = self.get_previous_states(self)

        for previous_state in previous_states:
            new_message = previous_state.to_message(verbose=False)
            if new_message and not content:
                content = new_message
            elif new_message:
                content += f"\n\n{new_message}"

            messages.append(UserMessage(content=content))

            if hasattr(previous_state.last_action.request, "action"):
                action = previous_state.last_action.request.action
            else:
                action = previous_state.last_action.request

            messages.append(
                AssistantMessage(
                    action=action
                )
            )
            content = ""

        content += self.to_message()

        file_context_str = self.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        messages.extend(self.retry_messages())

        return messages
