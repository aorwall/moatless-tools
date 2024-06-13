import fnmatch
import json
import logging
from typing import List, Type, Optional, Any, Dict

import litellm.utils
from pydantic import ValidationError, BaseModel, Field, root_validator, model_validator

from moatless.codeblocks import CodeBlockType
from moatless.file_context import FileContext, RankedFileSpan
from moatless.index.code_index import CodeIndex
from moatless.index.types import SearchCodeResponse
from moatless.loop.base import Loop, BaseState

from moatless.loop.prompt import (
    SEARCH_SYSTEM_PROMPT,
    SEARCH_FUNCTIONS_FEW_SHOT,
    FIND_AGENT_TEST_IGNORE,
)

from moatless.loop.base import Loop
from moatless.trajectory import Trajectory
from moatless.types import (
    FileWithSpans,
    ActionRequest,
    ActionSpec,
    Reject,
)
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class SearchCodeRequest(ActionRequest):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )
    query: Optional[str] = Field(
        default=None,
        description="A semantic similarity search query. Use natural language to describe what you are looking for.",
    )
    code_snippet: Optional[str] = Field(
        default=None,
        description="Specific code snippet to that should be exactly matched.",
    )
    class_name: Optional[str] = Field(
        default=None, description="Specific class name to include in the search."
    )
    function_name: Optional[str] = Field(
        default=None, description="Specific function name to include in the search."
    )

    @model_validator(mode="after")
    def check_at_least_one_field(self):
        if (
            not self.query
            and not self.code_snippet
            and not self.class_name
            and not self.function_name
        ):
            raise ValueError(
                "At least one of query, code_snippet, class_name, or function_name must be set"
            )
        return self


class SearchCodeAction(ActionSpec):

    code_index: CodeIndex

    @classmethod
    def request_class(cls) -> Type[SearchCodeRequest]:
        return SearchCodeRequest

    @classmethod
    def name(self) -> str:
        return "search"

    @classmethod
    def description(cls) -> str:
        return "Search for code."

    @classmethod
    def validate_request(cls, args: Dict[str, Any]) -> SearchCodeRequest:
        return cls.request_class().model_validate(args, strict=True)

    def __init__(self, code_index: CodeIndex):
        super().__init__(code_index=code_index)

    def search(self, request: SearchCodeRequest) -> SearchCodeResponse:
        return self.code_index.search(**request.dict())

    class Config:
        arbitrary_types_allowed = True


class IdentifyCodeRequest(ActionRequest):
    reasoning: str = Field(None, description="The reasoning for the code selection.")

    files_with_spans: List[FileWithSpans] = Field(
        default=None, description="The files and spans to select."
    )


class IdentifyCode(ActionSpec):

    @classmethod
    def request_class(cls):
        return IdentifyCodeRequest

    @classmethod
    def name(self):
        return "identify"

    @classmethod
    def description(cls) -> str:
        return "Identify the relevant code files and spans."


class FindCodeRequest(ActionRequest):
    instructions: Optional[str] = Field(
        default=None, description="Instructions to find code based on."
    )


class FindCodeResponse(BaseModel):
    message: Optional[str] = Field(None, description="A message to show the user.")
    files: list[FileWithSpans] = Field(
        default_factory=list, description="The files and spans found."
    )


class ActionCallWithContext(BaseModel):
    call_id: str
    action_name: str
    arguments: dict
    file_context: FileContext
    message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class Searching(BaseState):

    def actions(self) -> list[Type[ActionSpec]]:
        return [SearchCodeAction, IdentifyCode, Reject]


class SearchLoop(Loop):

    def __init__(
        self,
        workspace: Workspace,
        instructions: str,
        trajectory: Optional[Trajectory] = None,
        do_initial_search: bool = False,
        max_context_size: int = 4000,
        token_decay_rate: float = 1.2,
        combined_function: bool = True,
        support_test_files: bool = False,
        **kwargs,
    ):
        super().__init__(trajectory=trajectory, **kwargs)

        # TODO: Move all instance vars to pydantic fields and private attrs
        self._workspace = workspace
        self._tool_calls: list[ActionCallWithContext] = []
        self._instructions = instructions

        self._trajectory = trajectory or workspace.create_trajectory(
            "code_finder", input_data={"instructions": instructions}
        )
        self._max_context_size = max_context_size
        self._token_decay_rate = token_decay_rate
        self._do_initial_search = do_initial_search
        self._support_test_files = support_test_files
        self._combined_function = combined_function

        self.transition(Searching())

        system_prompt = SEARCH_SYSTEM_PROMPT + SEARCH_FUNCTIONS_FEW_SHOT
        if not self._support_test_files:
            system_prompt += FIND_AGENT_TEST_IGNORE

        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}

        self._initial_messages = [system_message, instruction_message]

        self._is_running = False
        self._is_retry = False
        self._previous_arguments: dict = {}
        self._retry_messages: list[dict] = []

    @classmethod
    def name(cls) -> str:
        return "find_code"

    @classmethod
    def description(cls) -> str:
        return "An atonomous agent used to find code based on a software requirement."

    @classmethod
    def request_class(cls) -> Type[ActionRequest]:
        return FindCodeRequest

    def loop(self, response_message: litellm.Message) -> Optional[FindCodeResponse]:
        self._retry_messages = []

        if response_message.content:
            self._workspace.save_trajectory_thought(response_message.content)
            logger.info(f"Thought: {response_message.content}")

        if hasattr(response_message, "tool_calls"):
            for tool_call in response_message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse arguments: {tool_call.function.arguments}"
                    )
                    return self._retry(
                        response_message,
                        f"Failed to parse arguments: {tool_call.function.arguments}. Make sure the function is called properly.",
                        tool_call=tool_call,
                    )

                function_name = tool_call.function.name
                if function_name == IdentifyCode.name():
                    response = self._identify(tool_call.id, arguments)
                    if response:
                        return response
                elif function_name == Reject.name():
                    reject_request = Reject.model_validate(arguments)
                    return FindCodeResponse(message=reject_request.reason)
                elif function_name == SearchCodeAction.name():
                    ranked_spans = []
                    try:
                        if self._previous_arguments == arguments:
                            logger.warning(
                                f"Got same arguments as last call: {arguments}.."
                            )
                            if self._is_retry:
                                raise Exception(
                                    f"Got same arguments as last call to {function_name} and {arguments}."
                                )
                            message = "The search arguments are the same as the previous call. You must use different arguments to continue."
                        else:
                            self._previous_arguments = arguments

                            search_action = SearchCodeAction(self._workspace.code_index)
                            search_request = search_action.validate_request(arguments)

                            if (
                                not self._support_test_files
                                and search_request.file_pattern
                                and is_test_pattern(search_request.file_pattern)
                            ):
                                message = "It's not possible to search for test files."
                            else:
                                search_result = search_action.search(search_request)

                                for hit in search_result.hits:
                                    for span in hit.spans:
                                        ranked_spans.append(
                                            RankedFileSpan(
                                                file_path=hit.file_path,
                                                span_id=span.span_id,
                                                rank=span.rank,
                                            )
                                        )
                                message = search_result.message
                    except ValidationError as e:
                        logger.warning(f"Failed to validate function call. Error: {e}")
                        message = f"The function call is invalid. Error: {e}"

                        if self._is_retry:
                            raise e

                        self._is_retry = True
                    except Exception as e:
                        raise e

                    self._add_to_message_history(
                        tool_call.id,
                        function_name,
                        arguments,
                        ranked_spans,
                        message,
                    )
                    return None
                else:
                    logger.warning(f"Unknown function used: {function_name}")
                    return self._retry(
                        response_message,
                        f"Unknown function: {function_name}",
                        tool_call=tool_call,
                    )

                self._is_retry = False

        elif self._is_retry:
            logger.warning(f"The LLM retried without a tool call, aborting")
            raise Exception("The LLM retried without a tool call, aborting")
        elif self._tool_calls:
            return self._retry(
                response_message,
                f"I expected a function call in the response. If you're done, please use the identify function.",
            )
        else:
            return self._retry(
                response_message, f"I expected a function call in the response."
            )

        return None

    def _retry(
        self,
        response_message,
        message: str,
        tool_call: Optional[litellm.utils.ChatCompletionMessageToolCall] = None,
    ):
        self._retry_messages.append(response_message.dict())

        if tool_call:
            self._retry_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": message,
                }
            )
        else:
            self._retry_messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )
        self._is_retry = True

    def _identify(
        self, tool_call_id: str, arguments: dict[str, Any]
    ) -> Optional[FindCodeResponse]:
        identify_request = IdentifyCodeRequest.model_validate(arguments)

        self._trajectory.save_action(
            IdentifyCode.name(),
            input=identify_request.dict(),
            output={},
        )

        files_with_missing_spans = self._find_missing_spans(
            identify_request.files_with_spans
        )
        if files_with_missing_spans:
            message = "identify() The following span ids are not in the file context: "

            for file_with_spans in files_with_missing_spans:
                message += f"\n{file_with_spans.file_path}: {', '.join(file_with_spans.span_ids)}"

            logger.warning(message)

            # self._add_to_message_history(
            #    tool_call_id, "identify", arguments, [], message
            # )

            # return None

        file_context = self._workspace.create_file_context()
        for file_with_spans in identify_request.files_with_spans:
            file_context.add_spans_to_context(
                file_with_spans.file_path,
                file_with_spans.span_ids,
            )

        logger.info(
            f"find_code: Found {len(file_context.files)} files and {file_context.context_size()}."
        )

        # self._expand_context_with_related_spans(file_context)

        response = FindCodeResponse(
            message=identify_request.reasoning, files=file_context.to_files_with_spans()
        )

        self._trajectory.save_output(response.dict())

        return response

    def _add_to_message_history(
        self,
        call_id: str,
        action_name: str,
        arguments: dict,
        ranked_spans: List[RankedFileSpan],
        message: Optional[str] = None,
    ):
        file_context = self._workspace.create_file_context()
        file_context.add_ranked_spans(ranked_spans)

        for previous_call in self._tool_calls:
            for span in ranked_spans:
                previous_call.file_context.remove_span_from_context(
                    span.file_path, span.span_id, remove_file=True
                )

        self._tool_calls.append(
            ActionCallWithContext(
                call_id=call_id,
                action_name=action_name,
                arguments=arguments,
                file_context=file_context,
                message=message,
            )
        )

        self._trajectory.save_action(
            action_name,
            input=arguments,
            output={
                "file_context": file_context.dict(),
                "message": message,
            },
        )

    def message_history(self) -> list[dict]:
        messages = []
        for tool_call in self._tool_calls:
            arguments_json = (
                json.dumps(tool_call.arguments) if tool_call.arguments else "{}"
            )
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call.action_name,
                                "arguments": arguments_json,
                            },
                        }
                    ],
                }
            )

            content = tool_call.message or ""
            if tool_call.file_context:
                content += "\n\n"
                content += tool_call.file_context.create_prompt(
                    show_span_ids=True,
                    show_line_numbers=False,
                    exclude_comments=True,
                    show_outcommented_code=True,
                    outcomment_code_comment="... rest of the code",
                )

            messages.append(
                {
                    "tool_call_id": tool_call.call_id,
                    "role": "tool",
                    "name": tool_call.action_name,
                    "content": content,
                }
            )

        return self._initial_messages + messages + self._retry_messages

    def _find_missing_spans(self, files_with_spans: list[FileWithSpans]):
        files_with_missing_spans = []
        for file_with_spans in files_with_spans:
            missing_spans = []
            for span_id in file_with_spans.span_ids:
                if not self._span_is_in_context(file_with_spans.file_path, span_id):
                    missing_spans.append(span_id)

            if missing_spans:
                files_with_missing_spans.append(
                    FileWithSpans(
                        file_path=file_with_spans.file_path, span_ids=missing_spans
                    )
                )

        return files_with_missing_spans

    def _span_is_in_context(self, file_path: str, span_id: str) -> bool:
        for previous_call in self._tool_calls:
            if previous_call.file_context.has_span(file_path, span_id):
                return True

        return False

    def _expand_context_with_related_spans(self, file_context: FileContext):
        spans = 0

        # Add related spans if context allows it
        if file_context.context_size() > self._max_context_size:
            return spans

        for file in file_context.files:
            if not file.spans:
                continue

            related_span_ids = []
            for span_id in file.span_ids:
                span = file.module.find_span_by_id(span_id)

                if span.initiating_block.type == CodeBlockType.CLASS:
                    child_span_ids = span.initiating_block.get_all_span_ids()
                    for child_span_id in child_span_ids:
                        if self._span_is_in_context(file.file_path, child_span_id):
                            related_span_ids.append(child_span_id)

                related_span_ids.extend(file.module.find_related_span_ids(span_id))

                for related_span_id in related_span_ids:
                    if related_span_id in file.span_ids:
                        continue

                    related_span = file.module.find_span_by_id(related_span_id)
                    if (
                        related_span.tokens + file_context.context_size()
                        > self._max_context_size
                    ):
                        return spans

                    spans += 1
                    file.add_span(related_span_id)

        if spans > 0:
            logger.info(
                f"find_code: Expanded context with {spans} spans to {file_context.context_size()} tokens."
            )

    def _tool_arguments(
        self, tool_call: litellm.utils.ChatCompletionMessageToolCall
    ) -> dict[str, Any]:
        try:
            return json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse arguments: {tool_call.function.arguments}")
            self._trajectory.save_error(
                f"No Failed to parse argument: {tool_call.function.arguments}"
            )
            raise Exception(
                f"Failed to parse arguments: {tool_call.function.arguments}"
            )


def is_test_pattern(file_pattern: str):
    test_patterns = ["test_*.py", "/tests/"]
    for pattern in test_patterns:
        if pattern in file_pattern:
            return True

    if file_pattern.startswith("test"):
        return True

    test_patterns = ["test_*.py"]

    for pattern in test_patterns:
        if fnmatch.filter([file_pattern], pattern):
            return True

    return False
