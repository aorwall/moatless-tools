import json
import logging
import random
import string
from typing import Optional, List, Type, Any

import litellm
from pydantic import BaseModel, Field

from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.codeblocks import CodeBlockTypeGroup, BlockSpan
from moatless.loop.prompt import CODER_SYSTEM_PROMPT, SEARCH_REPLACE_PROMPT
from moatless.loop.base import BaseLoop, LoopState
from moatless.settings import Settings
from moatless.trajectory import Trajectory
from moatless.types import (
    Finish,
    ActionSpec,
    FileWithSpans,
    ActionRequest,
    Reject,
    RejectRequest,
    FinishRequest,
)
from moatless.utils.tokenizer import count_tokens
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class CodeResponse(BaseModel):
    message: str
    diff: Optional[str] = None


class RequestForChangeRequest(ActionRequest):
    description: str = Field(..., description="Description of the code change.")
    file_path: str = Field(..., description="The file path of the code to be updated.")
    span_id: str = Field(..., description="The span id of the code to be updated.")


class RequestForChange(ActionSpec):

    @classmethod
    def request_class(cls):
        return RequestForChangeRequest

    @classmethod
    def name(self):
        return "request_for_change"

    @classmethod
    def description(cls) -> str:
        return "Request for permission to change code."


class LineNumberClarificationRequest(ActionRequest):
    thoughts: str = Field(..., description="Thoughts on which lines to select")
    start_line: int = Field(
        ..., description="The start line of the code to be updated."
    )

    end_line: int = Field(..., description="The end line of the code to be updated.")


class LineNumberClarification(ActionSpec):

    @classmethod
    def request_class(cls):
        return LineNumberClarificationRequest

    @classmethod
    def name(self):
        return "specify_lines"

    @classmethod
    def description(cls) -> str:
        return "Specify which lines to change."


class Discard(ActionSpec):

    @classmethod
    def name(self):
        return "discard"

    @classmethod
    def description(cls) -> str:
        return "Discard changes."


class Save(ActionSpec):

    @classmethod
    def name(self):
        return "save"

    @classmethod
    def description(cls) -> str:
        return "Save changes."


class Pending(LoopState):

    def tools(self) -> list[Type[ActionSpec]]:
        return [RequestForChange, Finish, Reject]


class Clarification(LoopState):
    description: str
    file_path: str
    action: Type[ActionSpec]
    span_id: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    def tools(self) -> list[Type[ActionSpec]]:
        return [LineNumberClarification, Reject]


def change_to_tool_call(
    call_id: str, description: str, file_path: str, span_id: str
) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": RequestForChange.name(),
            "arguments": json.dumps(
                {"description": description, "file_path": file_path, "span_id": span_id}
            ),
        },
    }


class PreviousChange(BaseModel):
    description: str
    file_path: str
    span_id: str
    code_to_replace: Optional[str] = None
    code_replacement: Optional[str] = None
    diff: Optional[str] = None
    rejected: Optional[str] = None

    def to_tool_call(self, call_id: str) -> dict[str, Any]:
        return change_to_tool_call(
            call_id, self.description, self.file_path, self.span_id
        )

    def as_messages(self, function_call: bool = False):
        if self.rejected:
            content = f"The request was rejected with the reason: {self.rejected}"
        elif self.diff and function_call:
            content = f"The request was approved and implemented with the following diff:\n\n```diff\n{self.diff}\n```"
        elif self.diff:
            content = f"<replace>\n{self.code_replacement}\n</replace>"
        else:
            content = "No changes made."

        if function_call:
            fake_tool_call_id = generate_call_id()

            return [
                {
                    "tool_calls": [self.to_tool_call(fake_tool_call_id)],
                    "role": "assistant",
                },
                {
                    "tool_call_id": fake_tool_call_id,
                    "role": "tool",
                    "name": RequestForChange.name(),
                    "content": content,
                },
            ]
        else:
            return [
                {
                    "content": f"{self.description}\n\n<search>\n{self.code_to_replace}\n</search>",
                    "role": "user",
                },
                {
                    "content": content,
                    "role": "assistant",
                },
            ]


class CodeChange(LoopState):
    description: str
    file_path: str
    span_id: str
    start_line: int
    end_line: int
    code_to_replace: str
    code_replacement: Optional[str] = None
    diff: Optional[str] = None
    retry: int = 0

    def tools(self) -> list[Type[ActionSpec]]:
        return [Reject]

    def stop_words(self):
        return ["</replace>"]


class CodeLoop(BaseLoop):

    def __init__(
        self,
        workspace: Workspace,
        instructions: str,
        files: List[FileWithSpans],
        trajectory: Optional[Trajectory] = None,
        max_tokens_to_edit: int = 750,
        **kwargs,
    ):
        super().__init__(trajectory=trajectory, **kwargs)
        self._workspace = workspace
        self._instructions = instructions

        self._trajectory: Trajectory = trajectory or workspace.create_trajectory(
            "code_finder", input_data={"instructions": instructions}
        )

        self._file_context = self._workspace.create_file_context(files)
        self._file_context.expand_context_with_related_spans()
        self._file_context.expand_context_with_imports()
        # self._file_context.expand_small_classes(max_tokens_to_edit)

        self._previous_changes: List[PreviousChange] = []

        self._messages = []
        self.transition_to_pending()

        self._max_tokens_to_edit = max_tokens_to_edit

        self._is_running = False
        self._is_retry = False

    def loop(self, message: litellm.Message) -> Optional[CodeResponse]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            if message.content:
                self._workspace.save_trajectory_thought(message.content)
                logger.info(f"Thought: {message.content}")

            if len(message.tool_calls) > 1:
                logger.info(
                    "Multiple tool calls in one message, will only handle the first. The rest will be ignored."
                )

            tool_call = message.tool_calls[0]
            self._messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [tool_call],
                }
            )

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse arguments: {tool_call.function.arguments}"
                )
                self._trajectory.save_error(
                    f"Failed to parse arguments: {tool_call.function.arguments}"
                )
                self._messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": f"Failed to parse function call. Try again. Error: {tool_call.function.arguments}",
                    }
                )
                return None

            if tool_call.function.name == RequestForChange.name():
                request_for_change = RequestForChangeRequest.model_validate(arguments)
                logger.info(
                    f"Request for change in file: {request_for_change.file_path} with span: {request_for_change.span_id}"
                )

                span, response = self.verify_span(
                    request_for_change.file_path,
                    request_for_change.span_id,
                    check_parent=True,
                )

                if span and span.tokens > self._max_tokens_to_edit:
                    logger.info(
                        f"Span has {span.tokens} tokens, which is higher than the maximum allowed {self._max_tokens_to_edit} tokens. Ask for clarification."
                    )
                    self._messages = []
                    self.transition(
                        Clarification(
                            description=request_for_change.description,
                            file_path=request_for_change.file_path,
                            action=LineNumberClarification,
                            span_id=request_for_change.span_id,
                        )
                    )

                    self._trajectory.save_action(
                        tool_call.function.name,
                        input=arguments,
                        output={"response": response},
                    )
                    return None

                if response:
                    self._messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": response,
                        }
                    )

                else:
                    response = self.initiate_code_change(
                        request_for_change.description,
                        request_for_change.file_path,
                        request_for_change.span_id,
                    )

                self._trajectory.save_action(
                    tool_call.function.name,
                    input=arguments,
                    output={"response": response},
                )

            elif tool_call.function.name == LineNumberClarification.name():
                line_numbers = LineNumberClarificationRequest.model_validate(arguments)
                logger.info(
                    f"Got line number clarification: {line_numbers.start_line} -{line_numbers.end_line}"
                )
                if not isinstance(self.state, Clarification):
                    raise ValueError("Unexpected state.")

                response = self._verify_line_numbers(line_numbers)
                if response:
                    self._messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": response,
                        }
                    )

                    self._trajectory.save_action(
                        tool_call.function.name,
                        input=arguments,
                        output={"response": response},
                    )

                    return None

                response = self.initiate_code_change(
                    self.state.description,
                    self.state.file_path,
                    self.state.span_id,
                    line_numbers.start_line,
                    line_numbers.end_line,
                )

                self._trajectory.save_action(
                    tool_call.function.name,
                    input=arguments,
                    output={"response": response},
                )

            elif tool_call.function.name == Reject.name():
                rejection = RejectRequest.model_validate(arguments)
                logger.info(f"Rejecting with the reason: {rejection.reason}")
                self._workspace.trajectory.save_action(
                    name="reject", input={"reason": rejection.reason}
                )
                if isinstance(self.state, Pending):
                    return CodeResponse(message=rejection.reason)
                else:
                    rejected_change = PreviousChange(**self.state.model_dump())
                    rejected_change.rejected = rejection.reason

                    for previous_change in self._previous_changes:
                        if (
                            previous_change.rejected
                            and previous_change.file_path == rejected_change.file_path
                            and previous_change.span_id == rejected_change.span_id
                        ):
                            logger.warning(
                                f"A change to file {rejected_change.file_path} and span {rejected_change.span_id} already rejected. Will finish loop."
                            )
                            self._trajectory.save_output({"message": rejection.reason})
                            self._workspace.save()  # TODO: Should probably not save...
                            return CodeResponse(message=rejection.reason)

                    self._previous_changes.append(rejected_change)
                    self.transition_to_pending()

            elif tool_call.function.name == Finish.name():
                finish = FinishRequest.model_validate(arguments)

                logger.info(f"Finishing with the reason: {finish.reason}")
                self._workspace.trajectory.save_action(
                    name="finish", input={"reason": finish.reason}
                )

                # TODO: Add diff to response and trajectory
                self._trajectory.save_output({"message": finish.reason})

                self._workspace.save()
                return CodeResponse(message=finish.reason)
            else:
                logger.warning(f"Unknown function: {tool_call.function.name}")
                self._trajectory.save_error(
                    f"Unknown function: {tool_call.function.name}"
                )
                self._messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": f"Unknown function: {tool_call.function.name}",
                    }
                )
        elif isinstance(self.state, CodeChange):
            self._handle_code_change(message.content)
        else:
            if len(self._previous_changes) > 0:
                content = "Expected a function call. Use the finish function if you're finished with the changes. "
            else:
                content = "Expected a function call. Use the request_for_change function to request a change. "

            self._messages.append({"role": "user", "content": content})

        return None

    def transition_to_pending(self):
        self._messages = []
        self.transition(Pending())

    def verify_span(
        self, file_path: str, span_id: str, check_parent: bool = False
    ) -> [Optional[BlockSpan], Optional[str]]:
        context_file = self._file_context.get_file(file_path)
        if not context_file:
            return (
                None,
                f"File {file_path} is not found in the file context. You can only request changes to files that are in file context. ",
            )

        span = context_file.get_span(span_id)
        if not span:
            spans = self._file_context.get_spans(file_path)
            if not spans:
                raise ValueError(f"Spans not found for file: {file_path}")

            span_ids = [span.span_id for span in spans]

            if check_parent:
                span_not_in_context = context_file.file.module.find_span_by_id(span_id)

                # Check if the LLM is referring to a parent span shown in the prompt
                if (
                    span_not_in_context
                    and span_not_in_context.initiating_block.has_any_span(set(span_ids))
                ):
                    logger.info(
                        f"Use span {span_id} as it's a parent span of a span in the context."
                    )
                    span = span_not_in_context

            if not span:
                span_str = ", ".join(span_ids)
                logger.warning(
                    f"Span not found: {span_id}. Available spans: {span_str}"
                )
                return None, f"Span not found: {span_id}. Available spans: {span_str}"

        return span, None

    def initiate_code_change(
        self,
        description: str,
        file_path: str,
        span_id: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        logger.info(
            f"Initiating code change for span {span_id} in file {file_path}. Start line: {start_line}, end line: {end_line}"
        )

        context_file = self._file_context.get_file(file_path)
        file = context_file.file

        if start_line is None:
            span = file.module.find_span_by_id(span_id)
            edit_block_start_line = span.start_line
            edit_block_end_line = span.end_line
        else:
            edit_block_start_line, edit_block_end_line = file.get_line_span(
                start_line, end_line
            )

        code_lines = file.content.split("\n")
        lines_to_replace = code_lines[edit_block_start_line - 1 : edit_block_end_line]

        edit_block_code = "\n".join(lines_to_replace)

        tokens = count_tokens(edit_block_code)
        if tokens > self.state.max_tokens():
            raise ValueError(
                f"Span {span_id} in {file_path} ({edit_block_start_line} - {edit_block_end_line}) has {tokens} tokens, which is higher than the maximum allowed {self.state.max_tokens} tokens in completion."
            )
        elif tokens > self._max_tokens_to_edit:
            logger.warning(
                f"Span {span_id} in {file_path} ({edit_block_start_line} - {edit_block_end_line}) has {tokens} tokens, which is higher than the maximum allowed {self._max_tokens_to_edit} tokens. Will continue anyway..."
            )

        self.transition(
            CodeChange(
                description=description,
                code_to_replace=edit_block_code,
                file_path=file_path,
                span_id=span_id,
                start_line=edit_block_start_line,
                end_line=edit_block_end_line,
            )
        )
        self._messages = []

        logger.info(
            f"Initiated edit block for span {span_id} in file {file_path} from line {edit_block_start_line} to {edit_block_end_line}"
        )

        return edit_block_code

    def _handle_code_change(self, content: str):
        msg_split = content.split("<replace>")
        if len(msg_split) == 1:
            if not self._add_prepared_response:
                logger.warning(
                    f"No <replace> tag found in response without prepped tag: {msg_split[0]}"
                )
                response_message = {
                    "role": "user",
                    "content": f"You did not provide any code in the replace tag. If you want to reject the instructions, use the reject function.",
                }

                self._messages.append(response_message)
                return

            replacement_code = msg_split[0]
        else:
            if msg_split[0]:
                thought = msg_split[0]
                self._trajectory.save_thought(thought)

            replacement_code = msg_split[1]

        if not isinstance(self.state, CodeChange):
            raise ValueError(f"Unexpected state: {self.state}")

        context_file = self._file_context.get_file(self.state.file_path)
        update_result = context_file.update_content_by_line_numbers(
            self.state.start_line - 1, self.state.end_line, replacement_code
        )

        self._trajectory.save_action(
            "search_replace",
            input={
                "file_path": self.state.file_path,
                "span_id": self.state.span_id,
                "start_line": self.state.start_line,
                "end_line": self.state.end_line,
                "replacement_code": replacement_code,
            },
            output={
                "diff": update_result.diff,
                "updated": update_result.updated,
                "error": update_result.error,
                "new_span_ids": update_result.new_span_ids,
            },
        )

        if update_result.diff and update_result.updated:
            logger.info(
                f"Updated file {self.state.file_path} with diff:\n{update_result.diff}"
            )
            self.state.code_replacement = replacement_code
            self.state.diff = update_result.diff
            self._previous_changes.append(PreviousChange(**self.state.model_dump()))
            self.transition_to_pending()
            return

        if self.state.retry > 2:
            logger.warning(
                f"Failed after {self.state.retry} retries. Will reject change."
            )
            change = PreviousChange(**self.state.model_dump())
            change.rejected = "Failed to apply changes."
            self.transition_to_pending()
        if update_result.diff:
            logger.warning(f"Diff was not applied:\n{update_result.diff}")
            response_message = {
                "role": "user",
                "content": f"The following diff was not applied:\n {update_result.diff}. \n"
                f"Errors:\n{update_result.error}\n"
                f"Make sure that you return the unchanged code in the replace tag exactly as it is. "
                f"If you want to reject the instructions, use the reject function.",
            }
            self.state.retry += 1
        else:
            logger.info(f"No changes found in {self.state.file_path}.")
            response_message = {
                "role": "user",
                "content": f"The code in the replace tag is the same as in the search. Use the reject function if you can't do any changes and want to reject the instructions.",
            }
            self.state.retry += 1

        self._messages.append(response_message)

    def _verify_line_numbers(self, line_numbers: LineNumberClarificationRequest):
        context_file = self._file_context.get_file(self.state.file_path)
        span = context_file.file.module.find_span_by_id(self.state.span_id)

        logger.info(
            f"Verifying line numbers: {line_numbers.start_line} - {line_numbers.end_line}. To span with line numbers: {span.start_line} - {span.end_line}"
        )

        if (
            line_numbers.start_line <= span.start_line
            and line_numbers.end_line >= span.end_line
        ):
            return (
                f"The provided line numbers {line_numbers.start_line} - {line_numbers.end_line} are for the full code span. You must specify line numbers of only lines you want to change.",
            )

        logger.info(
            f"Checking if the span is a class/function signature to {span.initiating_block.path_string()} ({span.initiating_block.start_line} - {span.initiating_block.next.start_line})"
        )
        if (
            span.initiating_block.start_line
            == span.initiating_block.next.start_line
            < line_numbers.end_line
            and span.initiating_block.sum_tokens() > self._max_tokens_to_edit
        ):
            logger.info(
                f"The line numbers {line_numbers.start_line} - {line_numbers.end_line} only points to the signature of the {span.initiating_block.type.value}."
            )
            return f"The line numbers {line_numbers.start_line} - {line_numbers.end_line} only points to the signature of the {span.initiating_block.type.value}. You need to specify the exact part of the code that needs to be updated to fulfill the change."

        # TODO: Refactor duplicated code
        edit_block_start_line, edit_block_end_line = context_file.file.get_line_span(
            line_numbers.start_line, line_numbers.end_line
        )

        code_lines = context_file.file.content.split("\n")
        lines_to_replace = code_lines[edit_block_start_line - 1 : edit_block_end_line]

        edit_block_code = "\n".join(lines_to_replace)

        tokens = count_tokens(edit_block_code)
        if tokens > self.state.max_tokens():
            logger.info(
                f"Lines {edit_block_start_line} - {edit_block_end_line} has {tokens} tokens, which is higher than the maximum allowed {self.state.max_tokens} tokens in completion. Ask for clarification."
            )
            return f"Lines {edit_block_start_line} - {edit_block_end_line} has {tokens} tokens, which is higher than the maximum allowed {self.state.max_tokens} tokens in completion. You need to specify the exact part of the code that needs to be updated to fulfill the change."

        return None

    @property
    def _add_prepared_response(self):
        return Settings.coder.coding_model.startswith("claude")

    def message_history(self) -> list[dict]:
        messages = []
        if isinstance(self.state, CodeChange):
            system_prompt = SEARCH_REPLACE_PROMPT
        else:
            system_prompt = CODER_SYSTEM_PROMPT

        messages.append({"content": system_prompt, "role": "system"})
        messages.append({"content": self._instructions, "role": "user"})

        for previous_change in self._previous_changes:
            messages.extend(
                previous_change.as_messages(
                    function_call=not isinstance(self.state, CodeChange)
                )
            )

        if not isinstance(self.state, Clarification):
            show_span_ids = isinstance(self.state, Pending)
            exclude_comments = isinstance(self.state, Pending)

            file_context_str = self._file_context.create_prompt(
                show_span_ids=show_span_ids,
                exclude_comments=exclude_comments,
                show_outcommented_code=True,
            )

            if self._previous_changes:
                messages[-1]["content"] += (
                    "\n\nThis is the file context after the change: \n"
                    + file_context_str
                )
            else:
                messages[-1]["content"] += "\n\nFile context: \n" + file_context_str
        else:
            fake_tool_call_id = generate_call_id()
            messages.append(
                {
                    "tool_calls": [
                        change_to_tool_call(
                            fake_tool_call_id,
                            self.state.description,
                            self.state.file_path,
                            self.state.span_id,
                        )
                    ],
                    "role": "assistant",
                }
            )

            file = self._workspace.get_file(self.state.file_path)
            if not file:
                raise ValueError(f"File {self.state.file_path} not found.")

            span = file.module.find_span_by_id(self.state.span_id)
            if not span:
                raise ValueError(
                    f"Span {self.state.span_id} not found in file {self.state.file_path}"
                )

            file_context = self._workspace.create_file_context(
                [FileWithSpans(file_path=self.state.file_path, span_ids=[span.span_id])]
            )

            # Include all function/class signatures if the block is a class
            if span.initiating_block.type == CodeBlockType.CLASS:
                for child in span.initiating_block.children:
                    if (
                        child.type.group == CodeBlockTypeGroup.STRUCTURE
                        and child.belongs_to_span
                        and child.belongs_to_span.span_id != span.span_id
                    ):
                        file_context.add_span_to_context(
                            file_path=self.state.file_path,
                            span_id=child.belongs_to_span.span_id,
                            tokens=1,
                        )  # TODO: Change so 0 can be set and mean "only signature"

            file_context_str = file_context.create_prompt(
                show_line_numbers=True,
                show_span_ids=False,
                exclude_comments=False,
                show_outcommented_code=True,
                outcomment_code_comment="... other code",
            )

            messages.append(
                {
                    "tool_call_id": fake_tool_call_id,
                    "role": "tool",
                    "name": RequestForChange.name(),
                    "content": f"The code span {self.state.span_id} in {self.state.file_path} is too large to edit. "
                    f"You need to specify the exact part of the code that needs to be updated to fulfill the change:\n"
                    f"{self.state.description}\n\n"
                    f"Use the function `specify_lines` to specify the lines. "
                    f"\n\n{file_context_str}",
                }
            )

        if isinstance(self.state, CodeChange):
            messages.append(
                {
                    "content": f"{self.state.description}\n\n<search>\n{self.state.code_to_replace}\n</search>",
                    "role": "user",
                }
            )

        messages.extend(self._messages)

        return messages


def generate_call_id():
    prefix = "call_"
    chars = string.ascii_letters + string.digits
    length = 24

    random_chars = "".join(random.choices(chars, k=length))

    random_string = prefix + random_chars

    return random_string
