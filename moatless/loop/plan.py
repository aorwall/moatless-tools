import logging
from typing import Any, Type, Optional, Union

from openai import BaseModel
from pydantic import Field, PrivateAttr

from moatless import Settings
from moatless.codeblocks import CodeBlockType
from moatless.loop.base import Rejected, Finished, InitialState
from moatless.loop.clarify import ClarifyCodeChange
from moatless.loop.edit import EditCode
from moatless.loop.prompt import CODER_SYSTEM_PROMPT
from moatless.loop.utils import generate_call_id
from moatless.types import (
    ActionSpec,
    Finish,
    Reject,
    ActionRequest,
    Message,
    AssistantMessage,
    UserMessage,
)

logger = logging.getLogger("PlanToCode")


class RequestForChange(ActionRequest):
    action_name: str = "apply_change"

    description: str = Field(..., description="Description of the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")
    span_id: str = Field(..., description="The span id of the code to be updated.")


class TakeAction(ActionRequest):
    action: Union[RequestForChange, Finish, Reject]


class PlanToCode(InitialState):

    message: str = Field(
        ...,
        description="Message to the coder",
    )

    _max_tokens_to_edit = PrivateAttr(default=Settings.coder.max_tokens_in_edit_prompt)

    def __init__(self, message: str, **data):
        super().__init__(message=message, include_message_history=True, **data)

    def handle_action(self, action: TakeAction) -> Optional[str]:
        if isinstance(action.action, RequestForChange):
            return self._request_for_change(action.action)
        elif isinstance(action.action, Reject):
            return self.transition_to(Rejected(reason=action.action.reason))
        elif isinstance(action.action, Finish):
            return self.transition_to(Finished(reason=action.action.thoughts))

        raise ValueError(f"Unknown request type: {action.action}")

    def action_type(self) -> Type[TakeAction]:
        return TakeAction

    def _request_for_change(self, rfc: RequestForChange) -> str:
        logger.info(
            f"request_for_change(file_path={rfc.file_path}, span_id={rfc.span_id})"
        )

        context_file = self.file_context.get_file(rfc.file_path)
        if not context_file:
            logger.warning(
                f"request_for_change: File {rfc.file_path} is not found in the file context."
            )
            return (
                f"File {rfc.file_path} is not found in the file context. "
                f"You can only request changes to files that are in file context."
            )

        span = context_file.get_span(rfc.span_id)
        if not span:
            spans = self.file_context.get_spans(rfc.file_path)
            span_ids = [span.span_id for span in spans]

            span_not_in_context = context_file.file.module.find_span_by_id(rfc.span_id)

            # Check if the LLM is referring to a parent span shown in the prompt
            if (
                span_not_in_context
                and span_not_in_context.initiating_block.has_any_span(set(span_ids))
            ):
                logger.info(
                    f"{self}: Use span {rfc.span_id} as it's a parent span of a span in the context."
                )
                span = span_not_in_context

            if not span:
                span_str = ", ".join(span_ids)
                logger.warning(
                    f"{self}: Span not found: {rfc.span_id}. Available spans: {span_str}"
                )
                return f"Span not found: {rfc.span_id}. Available spans: {span_str}"

        # If span is for a class block, consider the whole class
        if span.initiating_block.type == CodeBlockType.CLASS:
            tokens = span.initiating_block.sum_tokens()
            end_line = span.initiating_block.end_line
            logger.info(
                f"{self}: Span {rfc.span_id} is a class block. Consider the whole class ({span.initiating_block.start_line} - {end_line}) with {tokens} tokens."
            )
        else:
            tokens = span.tokens
            end_line = span.end_line

        if tokens > Settings.coder.max_tokens_in_edit_prompt:
            logger.info(
                f"{self}: Span has {tokens} tokens, which is higher than the maximum allowed "
                f"{Settings.coder.max_tokens_in_edit_prompt} tokens. Ask for clarification."
            )
            return self.transition_to(
                ClarifyCodeChange(
                    description=rfc.description,
                    file_path=rfc.file_path,
                    span_id=rfc.span_id,
                )
            )

        return self.transition_to(
            EditCode(
                description=rfc.description,
                file_path=rfc.file_path,
                span_id=rfc.span_id,
                start_line=span.start_line,
                end_line=end_line,
            )
        )

    def system_prompt(self) -> str:
        return CODER_SYSTEM_PROMPT

    def messages(self) -> list[Message]:
        messages = []

        trajectory_steps = self.trajectory.get_steps(str(self))

        for step in trajectory_steps:
            messages.append(UserMessage(content=step.transition_input["message"]))
            messages.append(
                AssistantMessage(
                    action=step.actions[-1].action,
                )
            )

        file_context_str = self.file_context.create_prompt(
            show_span_ids=True,
            exclude_comments=True,
            show_outcommented_code=True,
        )

        messages.append(
            Message(
                role="user",
                content=f"{self.message}\n\nFile context:\n{file_context_str}",
            )
        )

        return messages
