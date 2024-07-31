import logging
from typing import Type, Optional, List

from pydantic import Field, ConfigDict, PrivateAttr

from moatless.codeblocks import CodeBlockType
from moatless.edit.clarify import _get_post_end_line_index, _get_pre_start_line
from moatless.edit.prompt import (
    CODER_SYSTEM_PROMPT,
    SELECT_SPAN_SYSTEM_PROMPT,
    CODER_FINAL_SYSTEM_PROMPT,
)
from moatless.state import AgenticState
from moatless.types import (
    ActionRequest,
    ActionResponse,
    Message,
    UserMessage,
    AssistantMessage,
    CodeChange,
)
from moatless.verify.lint import VerificationError

logger = logging.getLogger("PlanToCode")


class IncludeSpan(ActionRequest):
    file_path: Optional[str] = Field(None, description="Find by file path.")
    class_name: Optional[str] = Field(None, description="Find by class name.")
    function_name: Optional[str] = Field(None, description="Find by function name.")


class ApplyChange(ActionRequest):
    """
    Request to apply a change to the code.
    """

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    action: str = Field(
        ...,
        description="The action to take, possible values are 'modify', 'review', 'include', 'finish', 'reject'",
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

    include_spans: Optional[List[IncludeSpan]] = Field(
        None, description="Find spans to include."
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


class ApplyChanges(ActionRequest):
    """
    Request to apply a change to the code.
    """

    scratch_pad: str = Field(..., description="Your thoughts on the code change.")

    action: str = Field(
        ...,
        description="The action to take, possible values are 'modify', 'review', 'include', 'finish', 'reject'",
    )

    changes: Optional[List[CodeChange]] = Field(
        None, description="The changes to apply."
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


class ReviewCode(AgenticState):
    message: Optional[str] = Field(
        None,
        description="Message to the coder",
    )

    # TODO: Move to a new state handling changes
    diff: Optional[str] = Field(
        None,
        description="The diff of a previous code change.",
    )

    max_prompt_file_tokens: int = Field(
        4000,
        description="The maximum number of tokens in the file context to show in the prompt.",
    )

    max_tokens_in_edit_prompt: int = Field(
        500,
        description="The maximum number of tokens in a span to show the edit prompt.",
    )

    allow_hallucinated_spans: bool = Field(
        False,
        description="Allow hallucinated spans to be used in the edit prompt.",
    )

    finish_on_review: bool = Field(
        False, description="Whether to finish the task if a review is requested."
    )

    finish_on_no_errors: bool = Field(
        False,
        description="Whether to finish the task if no verification errors are found.",
    )

    _verification_errors: List[VerificationError] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        message: Optional[str] = None,
        diff: Optional[str] = None,
        max_prompt_file_tokens: int = 4000,
        max_tokens_in_edit_prompt: int = 500,
        max_iterations: int = 8,
        include_message_history=True,
        **data,
    ):
        super().__init__(
            message=message,
            diff=diff,
            include_message_history=include_message_history,
            max_prompt_file_tokens=max_prompt_file_tokens,
            max_tokens_in_edit_prompt=max_tokens_in_edit_prompt,
            max_iterations=max_iterations,
            **data,
        )

    def init(self) -> Optional[ActionResponse]:
        self._verification_errors = self.workspace.verify()

        self.file_context.reset_verification_errors()

        for verification_error in self._verification_errors:
            logger.info(f"Verification error: {verification_error}")
            self.file_context.add_verification_error(verification_error)

        if self.finish_on_no_errors and not self._verification_errors:
            return ActionResponse.transition(
                trigger="finish", output={"message": "No errors to review."}
            )

        return None

    def handle_action(self, action: ApplyChange) -> ActionResponse:
        if action.action == "review":
            if self.diff and self.finish_on_review:
                logger.info(f"Review suggested after diff, will finish")
                return ActionResponse.transition(
                    trigger="finish", output={"message": "Finish on suggested review."}
                )
            else:
                return ActionResponse.retry(
                    "Review isn't possible. If the change is done you can finish or reject the task."
                )

        if action.include_spans:
            found_response = ""
            not_found_response = ""
            for include_span in action.include_spans:
                logger.info(
                    f"include_span(file_path={include_span.file_path}, class_name={include_span.class_name}, function_name={include_span.function_name})"
                )

                if not include_span.class_name and not include_span.function_name:
                    return ActionResponse.retry(
                        "You must provide either a class name or a function name or both."
                    )

                search_response = self.workspace.code_index.find_by_name(
                    class_names=[include_span.class_name],
                    function_names=[include_span.function_name],
                )
                if len(search_response.hits) == 1:
                    found_response += f" * {search_response.hits[0].file_path}\n"
                    for span in search_response.hits[0].spans:
                        self.file_context.add_span_to_context(
                            file_path=search_response.hits[0].file_path,
                            span_id=span.span_id,
                        )
                        found_response += f"   - {span}\n"
                elif len(search_response.hits) > 1 and include_span.file_path:
                    file_name = include_span.file_path.split("/")[-1]
                    for hit in search_response.hits:
                        if file_name in hit.file_path:
                            found_response += f" * {hit.file_path}\n"
                            for span in hit.spans:
                                self.file_context.add_span_to_context(
                                    file_path=hit.file_path,
                                    span_id=span.span_id,
                                )
                                found_response += f"   - {span}\n"
                else:
                    if include_span.file_path:
                        not_found_response += f"{include_span.file_path}"

                    if include_span.class_name:
                        not_found_response += f" class: {include_span.class_name}"

                    if include_span.function_name:
                        not_found_response += f" function: {include_span.function_name}"

            response = ""
            if found_response:
                response += f"Found the following spans:\n{found_response}"

            if not_found_response:
                response += (
                    f"\nCouldn't find the following spans:\n{not_found_response}"
                )

            return ActionResponse.retry(response)

        if action.finish:
            self.file_context.save()

            return ActionResponse.transition(
                trigger="finish", output={"message": action.finish}
            )
        elif action.reject:
            return ActionResponse.transition(
                trigger="reject", output={"message": action.reject}
            )

        elif action.file_path and action.span_id:
            return self._request_for_change(action)

        return ActionResponse.retry(
            "You must either provide an apply_change action or finish."
        )

    def action_type(self) -> Type[ApplyChange]:
        return ApplyChange

    def _request_for_change(self, rfc: ApplyChange) -> ActionResponse:
        logger.info(
            f"request_for_change(file_path={rfc.file_path}, span_id={rfc.span_id})"
        )

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
                f"You can only request changes to files that are in file context:\n{files_str}. You can try to add them by using the include_span action."
            )

        block_span = context_file.get_block_span(rfc.span_id)
        if not block_span and context_file.file.supports_codeblocks:
            spans = self.file_context.get_spans(rfc.file_path)
            span_ids = [span.span_id for span in spans]

            span_not_in_context = context_file.file.module.find_span_by_id(rfc.span_id)
            if span_not_in_context and self.allow_hallucinated_spans:
                logger.info(
                    f"{self}: Span {rfc.span_id} is not found in the context. Will add it."
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
                    f"{self}: Use span {rfc.span_id} as it's a parent span of a span in the context."
                )
                block_span = span_not_in_context

            if not block_span:
                span_str = ", ".join(span_ids)
                logger.warning(
                    f"{self}: Span not found: {rfc.span_id}. Available spans: {span_str}"
                )
                return ActionResponse.retry(
                    f"Span not found: {rfc.span_id}. Available spans: {span_str}"
                )

        # If span is for a class block, consider the whole class
        if block_span:
            start_line = block_span.start_line
            if block_span.initiating_block.type == CodeBlockType.CLASS:
                tokens = block_span.initiating_block.sum_tokens()
                end_line = block_span.initiating_block.end_line
                logger.info(
                    f"{self}: Span {rfc.span_id} is a class block. Consider the whole class ({block_span.initiating_block.start_line} - {end_line}) with {tokens} tokens."
                )
            else:
                tokens = block_span.tokens
                end_line = block_span.end_line

        else:
            span = context_file.get_span(rfc.span_id)
            if not span:
                spans = self.file_context.get_spans(rfc.file_path)
                span_ids = [span.span_id for span in spans]
                span_str = ", ".join(span_ids)
                return ActionResponse.retry(
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
                f"{self}: Span has {tokens} tokens, which is higher than the maximum allowed "
                f"{self.max_tokens_in_edit_prompt} tokens. Ask for clarification."
            )
            return ActionResponse.transition(
                trigger="edit_code",
                output={
                    "instructions": rfc.instructions,
                    "file_path": rfc.file_path,
                    "span_id": rfc.span_id,
                },
            )

        return ActionResponse.transition(
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
        return (
            CODER_SYSTEM_PROMPT + SELECT_SPAN_SYSTEM_PROMPT + CODER_FINAL_SYSTEM_PROMPT
        )

    def to_message(self) -> str:
        response_msg = ""

        if self.message:
            response_msg += self.message

        if self.diff:
            response_msg += f"\n\n<diff>\n{self.diff}\n</diff>"

        error_str = ""
        for verification_error in self._verification_errors:
            error_str += f" * {verification_error.code}: {verification_error.message} (file: {verification_error.file_path}, line {verification_error.line})\n"

        if error_str:
            response_msg += (
                f"\n\nThe following verification errors was found:\n\n{error_str}\n"
            )

        return response_msg

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        if self.loop.trajectory.initial_message:
            content = f"<main_objective>\n{self.loop.trajectory.initial_message}\n</main_objective>"
        else:
            content = ""

        previous_transitions = self.loop.get_previous_transitions(self)

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
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... rest of the code",
        )

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        messages.extend(self.retry_messages())

        return messages
