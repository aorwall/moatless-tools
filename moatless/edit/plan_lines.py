import logging
from typing import Type, Optional, List

from pydantic import Field, ConfigDict

from moatless.codeblocks.codeblocks import CodeBlockTypeGroup
from moatless.edit.clarify import _get_post_end_line_index, _get_pre_start_line
from moatless.edit.prompt import (
    CODER_SYSTEM_PROMPT,
    SELECT_LINES_SYSTEM_PROMPT,
    CODER_FINAL_SYSTEM_PROMPT,
)
from moatless.state import AgenticState
from moatless.types import (
    ActionRequest,
    ActionResponse,
    Message,
    UserMessage,
    AssistantMessage,
)
from moatless.utils.tokenizer import count_tokens
from moatless.verify.lint import VerificationError

logger = logging.getLogger("PlanToCode")


class ApplyChange(ActionRequest):
    """
    Request to apply a change to the code.
    """

    thoughts: str = Field(..., description="Your thoughts on the code change.")

    instructions: Optional[str] = Field(
        None, description="Instructions to do the code change."
    )
    file_path: Optional[str] = Field(
        None, description="The file path of the code to be updated."
    )
    start_line: Optional[int] = Field(
        None, description="The start line of the code to be updated."
    )
    end_line: Optional[int] = Field(
        None, description="The end line of the code to be updated."
    )

    reject: Optional[str] = Field(
        ..., description="Reject the request and explain why."
    )
    finish: Optional[str] = Field(
        None, description="Finish the request and explain why"
    )

    model_config = ConfigDict(
        extra="allow",
    )


class PlanToCodeWithLines(AgenticState):

    message: Optional[str] = Field(
        None,
        description="Message to the coder",
    )

    # TODO: Move to a new state handling changes
    diff: Optional[str] = Field(
        None,
        description="The diff of a previous code change.",
    )

    # TODO: Move to a new state handling lint problems
    verification_errors: Optional[List[VerificationError]] = Field(
        None,
        description="The verification errors from the previous code change.",
    )

    max_tokens_in_edit_prompt: int = Field(
        500,
        description="The maximum number of tokens in a span to show the edit prompt.",
    )

    expand_context_with_related_spans: bool = Field(
        True,
        description="Whether to expand the context with related spans.",
    )

    def __init__(
        self,
        message: Optional[str] = None,
        diff: Optional[str] = None,
        lint_messages: Optional[List[VerificationError]] = None,
        max_iterations: int = 5,
        **data,
    ):
        super().__init__(
            message=message,
            diff=diff,
            lint_messages=lint_messages,
            include_message_history=True,
            max_iterations=max_iterations,
            **data,
        )

    def init(self):
        # TODO: Make addition to context customizable??

        for error in self.verification_errors:
            self.file_context.add_file(
                file_path=error.file_path
            )  # TODO: BY line number!

        self.file_context.expand_context_with_init_spans()

        if (
            self.expand_context_with_related_spans
            and len(self.loop.trajectory.get_transitions(self.name)) == 0
        ):
            self.file_context.expand_context_with_related_spans(max_tokens=4000)

    def handle_action(self, action: ApplyChange) -> ActionResponse:
        if action.finish:
            self.file_context.save()

            return ActionResponse.transition(
                trigger="finish", output={"message": action.finish}
            )
        elif action.reject:
            return ActionResponse.transition(
                trigger="reject", output={"message": action.reject}
            )

        elif action.file_path:
            return self._request_for_change(action)

        return ActionResponse.retry(
            "You must either provide an apply_change action or finish."
        )

    def action_type(self) -> Type[ApplyChange]:
        return ApplyChange

    def _request_for_change(self, rfc: ApplyChange) -> ActionResponse:
        logger.info(f"request_for_change(file_path={rfc.file_path}")

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            logger.warning(
                f"request_for_change: File {rfc.file_path} is not found in the file context."
            )

            files_str = ""
            for file in self.file_context.files:
                files_str += f" * {file.file_path}\n"

            return ActionResponse.retry(
                f"File {rfc.file_path} is not found in the file context. "
                f"You can only request changes to files that are in file context:\n{files_str}"
            )

        if (
            not rfc.start_line
            and context_file.module.sum_tokens() > self.max_tokens_in_edit_prompt
        ):
            return ActionResponse.retry(
                f"The file {rfc.file_path} is to big to edit in one go, please provide start and end line numbers to specify the part of the code that needs to be updated."
            )

        block = context_file.module.find_first_by_start_line(rfc.start_line)

        if block.type.group == CodeBlockTypeGroup.STRUCTURE:
            structure_block = block
        else:
            structure_block = block.find_type_group_in_parents(
                CodeBlockTypeGroup.STRUCTURE
            )

        if structure_block.sum_tokens() < self.max_tokens_in_edit_prompt:
            return ActionResponse.transition(
                trigger="edit_code",
                output={
                    "instructions": rfc.instructions,
                    "file_path": rfc.file_path,
                    "start_line": structure_block.start_line,
                    "end_line": structure_block.end_line,
                },
            )

        last_structure_block_signature_line = structure_block.children[0].start_line - 1
        logger.info(
            f"{self}: Checking if the line numbers only covers a class/function signature to "
            f"{structure_block.path_string()} ({structure_block.start_line} - {last_structure_block_signature_line})"
        )
        if (
            rfc.start_line == block.start_line
            and last_structure_block_signature_line >= rfc.end_line
        ):
            clarify_msg = f"The line numbers {rfc.start_line} - {rfc.end_line} only covers to the signature of the {block.type.value}."
            logger.info(f"{self}: {clarify_msg}. Ask for clarification.")
            # TODO: Ask if this was intentional instead instructing the LLM
            return ActionResponse.retry(
                f"{clarify_msg}. You need to specify the exact part of the code that needs to be updated to fulfill the change."
            )

        code_lines = context_file.file.content.split("\n")
        lines_to_replace = code_lines[rfc.start_line - 1 : rfc.end_line]

        edit_block_code = "\n".join(lines_to_replace)

        tokens = count_tokens(edit_block_code)
        if tokens > self.max_tokens_in_edit_prompt:
            clarify_msg = f"Lines {rfc.start_line} - {rfc.end_line} has {tokens} tokens, which is higher than the maximum allowed {self.max_tokens_in_edit_prompt} tokens in completion"
            logger.info(f"{self} {clarify_msg}. Ask for clarification.")
            return ActionResponse.retry(
                f"{clarify_msg}. You need to specify the exact part of the code that needs to be updated to fulfill the change. If this is not possible you should reject the request."
            )

        start_line = _get_pre_start_line(
            rfc.start_line, structure_block.start_line, code_lines
        )
        end_line = _get_post_end_line_index(
            rfc.end_line, structure_block.end_line, code_lines
        )

        return ActionResponse.transition(
            trigger="edit_code",
            output={
                "instructions": rfc.instructions,
                "file_path": rfc.file_path,
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    def system_prompt(self) -> str:
        return (
            CODER_SYSTEM_PROMPT + SELECT_LINES_SYSTEM_PROMPT + CODER_FINAL_SYSTEM_PROMPT
        )

    def to_message(self) -> str:
        response_msg = ""

        if self.message:
            response_msg += self.message

        if self.diff:
            response_msg += f"\n\n<diff>\n{self.diff}\n</diff>"

        if self.verification_errors:
            lint_str = ""
            for lint_message in self.verification_errors:
                if lint_message.code[0] in ["E", "F"]:
                    lint_str += f" * {lint_message.code}: {lint_message.message} (line {lint_message.line})\n"

            if lint_str:
                response_msg += f"\n\nThe following lint errors was introduced after this change:\n<lint_errors>\n{lint_str}\n</lint_errors>"

        return response_msg

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        content = self.loop.trajectory.initial_message or ""

        previous_transitions = self.loop.trajectory.get_transitions(str(self))

        for transition in previous_transitions:

            new_message = transition.state.to_message()
            if new_message and not content:
                content = new_message
            elif new_message:
                content += f"\n\n{new_message}"

            messages.append(UserMessage(content=content))
            messages.append(
                AssistantMessage(
                    action=transition.actions[-1].action,
                )
            )
            content = ""

        content += self.to_message()
        file_context_str = self.file_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=True,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        messages.extend(self.retry_messages())

        return messages
