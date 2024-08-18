import logging
from typing import Optional

from pydantic import ConfigDict, Field, PrivateAttr

from moatless.codeblocks import CodeBlockType
from moatless.edit.clarify import _get_post_end_line_index, _get_pre_start_line
from moatless.edit.prompt import (
    CODER_FINAL_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    SELECT_SPAN_SYSTEM_PROMPT, WRITE_CODE_SUGGESTIONS_PROMPT,
)
from moatless.state import AgenticState, ActionRequest, StateOutcome, AssistantMessage, Message, UserMessage

from moatless.verify.lint import VerificationError

logger = logging.getLogger("PlanToCode")


class ApplyChange(ActionRequest):
    """
    Request to apply a change to the code.
    """

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    action: str = Field(
        ...,
        description="The action to take, possible values are 'modify', 'review', 'finish', 'reject'",
    )

    instructions: Optional[str] = Field(
        None, description="Instructions to do the code change."
    )
    file_path: Optional[str] = Field(
        None, description="The file path of the code to be updated."
    )
    span_id: Optional[str] = Field(
        None, description="The span id of the code to be updated."
    )

    reject: Optional[str] = Field(
        None, description="Reject the request and explain why."
    )
    finish: Optional[str] = Field(
        None, description="Finish the request and explain why"
    )

    model_config = ConfigDict(
        extra="allow",
    )


class PlanToCode(AgenticState):
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
    verification_errors: list[VerificationError] | None = Field(
        None,
        description="The lint errors of the previous code change.",
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    max_tokens_in_edit_prompt: int = Field(
        500,
        description="The maximum number of tokens in a span to show the edit prompt.",
    )

    expand_classes_with_max_tokens: Optional[int] = Field(
        None,
        description="The maximum number of tokens in a class to expand the context. If None, the context will not be expanded.",
    )

    expand_context_with_related_spans: bool = Field(
        True,
        description="Whether to expand the context with related spans.",
    )

    allow_hallucinated_spans: bool = Field(
        False,
        description="Whether to allow spans that exists but aren't found in the file context.",
    )

    finish_on_review: bool = Field(
        False, description="Whether to finish the task if a review is requested."
    )

    include_message_history: bool = Field(
        True,
        description="Whether to include the message history in the prompt.",
    )

    write_code_suggestions: bool = Field(
        True,
        description="Whether to instruct the LLM to write out the actual code in the instructions.",
    )

    _expanded_context: bool = PrivateAttr(False)

    def _execute_action(self, action: ApplyChange) -> StateOutcome:
        if action.action == "review":
            if self.diff and self.finish_on_review:
                logger.info("Review suggested after diff, will finish")
                return StateOutcome.transition(
                    trigger="finish", output={"message": "Finish on suggested review."}
                )
            else:
                return StateOutcome.retry(
                    "Review isn't possible. If the change is done you can finish or reject the task."
                )

        if action.action == "finish":
            return StateOutcome.transition(
                trigger="finish", output={"message": action.finish}
            )
        elif action.reject:
            return StateOutcome.transition(
                trigger="reject", output={"message": action.reject}
            )

        elif action.file_path and action.span_id:
            return self._request_for_change(action)

        return StateOutcome.retry(
            "You must either provide an apply_change action or finish."
        )

    def action_type(self) -> type[ApplyChange]:
        return ApplyChange

    def _request_for_change(self, rfc: ApplyChange) -> StateOutcome:
        logger.info(
            f"request_for_change(file_path={rfc.file_path}, span_id={rfc.span_id})"
        )

        if not rfc.instructions:
            return StateOutcome.retry(
                f"Please provide instructions for the code change."
            )

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            logger.warning(
                f"request_for_change: File {rfc.file_path} is not found in the file context."
            )

            files_str = ""
            for file in self.file_context.files:
                files_str += f" * {file.file_path}\n"

            return StateOutcome.retry(
                f"File {rfc.file_path} is not found in the file context. "
                f"You can only request changes to files that are in file context:\n{files_str}"
            )

        block_span = context_file.get_block_span(rfc.span_id)
        if not block_span and context_file.file.supports_codeblocks:
            spans = self.file_context.get_spans(rfc.file_path)
            span_ids = [span.span_id for span in spans]

            span_not_in_context = context_file.file.module.find_span_by_id(rfc.span_id)
            if span_not_in_context and self.allow_hallucinated_spans:
                logger.info(
                    f"{self.name}: Span {rfc.span_id} is not found in the context. Will add it."
                )
                block_span = span_not_in_context
                self.file_context.add_span_to_context(
                    file_path=rfc.file_path, span_id=block_span.span_id
                )

            # Check if the LLM is referring to a parent span shown in the prompt
            if (
                span_not_in_context
                and span_not_in_context.initiating_block.has_any_span(set(span_ids))
            ):
                logger.info(
                    f"{self.name}: Use span {rfc.span_id} as it's a parent span of a span in the context."
                )
                block_span = span_not_in_context

            if not block_span:
                span_str = ", ".join(span_ids)
                logger.warning(
                    f"{self.name}: Span not found: {rfc.span_id}. Available spans: {span_str}"
                )
                return StateOutcome.retry(
                    f"Span not found: {rfc.span_id}. Available spans: {span_str}"
                )

        # If span is for a class or function block, consider the whole span
        if block_span:
            tokens = block_span.initiating_block.sum_tokens()
            if (
                block_span.initiating_block.type
                in [CodeBlockType.CLASS, CodeBlockType.FUNCTION]
                and tokens < self.max_tokens_in_edit_prompt
            ):
                logger.info(
                    f"{self.name}: Span {rfc.span_id} is a {block_span.initiating_block.type} with {tokens} tokens. Return the whole block."
                )

                self.file_context.add_spans_to_context(
                    file_path=rfc.file_path,
                    span_ids=set(block_span.initiating_block.span_ids),
                )

                return StateOutcome.transition(
                    trigger="edit_code",
                    output={
                        "instructions": rfc.instructions,
                        "file_path": rfc.file_path,
                        "span_id": rfc.span_id,
                        "start_line": block_span.initiating_block.start_line,
                        "end_line": block_span.initiating_block.end_line,
                    },
                )
            else:
                start_line = block_span.start_line
                tokens = block_span.tokens
                end_line = block_span.end_line

        else:
            span = context_file.get_span(rfc.span_id)
            if not span:
                spans = self.file_context.get_spans(rfc.file_path)
                span_ids = [span.span_id for span in spans]
                span_str = ", ".join(span_ids)
                return StateOutcome.retry(
                    f"Span not found: {rfc.span_id}. Available spans: {span_str}"
                )

            content_lines = context_file.file.content.split("\n")
            start_line = _get_pre_start_line(span.start_line, 1, content_lines)
            end_line = _get_post_end_line_index(
                span.end_line, len(content_lines), content_lines
            )

            # TODO: Support token count in files without codeblock support
            tokens = 0

        if tokens > self.max_tokens_in_edit_prompt:
            logger.info(
                f"{self.name}: Span has {tokens} tokens, which is higher than the maximum allowed "
                f"{self.max_tokens_in_edit_prompt} tokens. Ask for clarification."
            )
            return StateOutcome.transition(
                trigger="edit_code",
                output={
                    "instructions": rfc.instructions,
                    "file_path": rfc.file_path,
                    "span_id": rfc.span_id,
                },
            )

        return StateOutcome.transition(
            trigger="edit_code",
            output={
                "instructions": rfc.instructions,
                "file_path": rfc.file_path,
                "span_id": rfc.span_id,
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    def system_prompt(self) -> str:
        if self.write_code_suggestions:
            return CODER_SYSTEM_PROMPT + WRITE_CODE_SUGGESTIONS_PROMPT + SELECT_SPAN_SYSTEM_PROMPT + CODER_FINAL_SYSTEM_PROMPT

        return (
            CODER_SYSTEM_PROMPT + SELECT_SPAN_SYSTEM_PROMPT + CODER_FINAL_SYSTEM_PROMPT
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
                lint_str += f" * {lint_message.code}: {lint_message.message} (line {lint_message.line})\n"

            if lint_str:
                response_msg += f"\n\nThe following lint errors was introduced after this change:\n<lint_errors>\n{lint_str}\n</lint_errors>"

        return response_msg

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        if self.initial_message:
            content = f"<issue>\n{self.initial_message}\n</issue>\n"
        else:
            content = ""

        previous_states = self.get_previous_states(self)

        for previous_state in previous_states:
            new_message = previous_state.to_message()
            if new_message and not content:
                content = new_message
            elif new_message:
                content += f"\n\n{new_message}"

            messages.append(UserMessage(content=content))
            messages.append(
                AssistantMessage(
                    action=previous_state.last_action.request,
                )
            )
            content = ""

        content += self.to_message()
        file_context_str = self.file_context.create_prompt(
            show_span_ids=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        messages.extend(self.retry_messages())

        return messages
