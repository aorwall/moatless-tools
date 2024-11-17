import logging
from abc import ABC
from typing import List, Optional, Type, Any, ClassVar, Tuple

from pydantic import Field, PrivateAttr, BaseModel

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation
from moatless.completion import CompletionModel
from moatless.completion.model import UserMessage, AssistantMessage, Completion
from moatless.exceptions import CompletionRejectError
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse
from moatless.repository.repository import Repository

logger = logging.getLogger(__name__)

IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with identifying relevant code in a codebase. Your goal is to select key code sections from the search results that are most relevant to the search request.

The previous messages will contain:
1. A search request from an AI assistant
2. Search results containing various code sections with their line numbers

# Your Task:

1. Understand the Search Request:
   * Analyze the previous search request to understand what code elements are being looked for
   * Identify key elements such as functions, variables, classes, or patterns that are relevant

2. Evaluate Search Results:
   * Examine each code section in the search results for alignment with the search request
   * Assess the relevance and importance of each code section
   * Consider the complete context of code sections

3. Respond with the Identify Action:
   * Select and respond with the code sections that best match the search request
   * Provide your analysis in the scratch_pad field
   * List the relevant file paths with start and end line numbers in the identified_spans field
"""


class SearchBaseArgs(ActionArguments, ABC):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific files or directories.",
    )


class IdentifiedSpans(BaseModel):
    file_path: str = Field(
        description="The file path where the relevant code is found."
    )
    start_line: int = Field(
        description="Starting line number of the relevant code section."
    )
    end_line: int = Field(
        description="Ending line number of the relevant code section."
    )


class Identify(ActionArguments):
    """Identify if the provided search result is relevant to the reported issue."""

    scratch_pad: str = Field(
        ...,
        description="Your thoughts and analysis on the search results and how they relate to the reported issue.",
    )

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code sections in the search results identified as relevant to the reported issue.",
    )


class SearchBaseAction(Action):
    args_schema: ClassVar[Type[ActionArguments]] = SearchBaseArgs

    max_search_tokens: int = Field(
        2000,
        description="The maximum number of tokens allowed in the search results.",
    )
    max_identify_tokens: int = Field(
        8000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_hits: int = Field(
        10,
        description="The maximum number of search hits to display.",
    )
    completion_model: Optional[CompletionModel] = Field(
        None,
        description="The completion model used to identify relevant code sections in search results.",
    )

    _repository: Repository = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()

    def __init__(
        self,
        repository: Repository | None = None,
        code_index: CodeIndex | None = None,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        super().__init__(completion_model=completion_model, **data)
        self._repository = repository
        self._code_index = code_index

    def execute(
        self, args: SearchBaseArgs, file_context: FileContext | None = None
    ) -> Observation:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the search action."
            )

        properties = {"search_hits": [], "search_tokens": 0}

        search_result_context = self._search_for_context(args)

        if search_result_context.is_empty():
            properties["fail_reason"] = "no_search_hits"
            return Observation(message="No search results found", properties=properties)

        properties["search_tokens"] = search_result_context.context_size()

        completion = None
        if (
            search_result_context.context_size() > self.max_search_tokens
            or search_result_context.span_count() > self.max_hits
        ):
            logger.info(
                f"{self.name}: Search too large. {properties['search_tokens']} tokens and {search_result_context.span_count()} hits, will ask for clarification."
            )
            view_context, completion = self._identify_code(args, search_result_context)
        else:
            view_context = search_result_context

        span_count = search_result_context.span_count()
        search_result_str = f"Found {span_count} code sections."

        if view_context.is_empty():
            search_result_str += (
                "\n\nNone of the search results was relevant to the task."
            )
            summary = "Didn't find any relevant code sections in the search results."
            message = search_result_str
        else:
            summary = "Found relevant code sections in the search results."
            search_result_str += "\n\nViewed relevant code:"
            message = (
                search_result_str
                + "\n"
                + view_context.create_prompt(
                    show_span_ids=False,
                    show_line_numbers=True,
                    exclude_comments=False,
                    show_outcommented_code=True,
                )
            )

        new_span_ids = file_context.add_file_context(view_context)
        properties["new_span_ids"] = new_span_ids

        logger.info(
            f"{self.name}: Found {span_count} code sections in search results. Viewed {view_context.span_count()} code sections."
        )

        return Observation(
            message=message,
            summary=summary,
            properties=properties,
            execution_completion=completion,
        )

    def _search_for_context(self, args: SearchBaseArgs) -> FileContext:
        search_result = self._search(args)
        if not search_result.hits:
            search_result = self._search_for_alternative_suggestion(args)
            logger.info(
                f"{self.name}: No relevant search results found. Will use alternative suggestion with {search_result.hits} hits."
            )

        span_count = 0
        search_result_context = FileContext(repo=self._repository)
        for hit in search_result.hits:
            span_count += len(hit.spans)
            for span in hit.spans:
                search_result_context.add_span_to_context(
                    hit.file_path, span.span_id, add_extra=True
                )

        return search_result_context

    def _select_span_instructions(self, search_result: SearchCodeResponse) -> str:
        if not self.add_to_context:
            return f"Here's the search result with the first line of codes in each code block. Use ViewCode to view specific code sections. "

        return f"The search result is too large. You must identify the relevant code sections in the search results to use them. "

    def _select_span_response_prompt(self, search_result: SearchCodeResponse) -> str:
        search_result_context = FileContext(repo=self._repository)
        for hit in search_result.hits:
            for span in hit.spans:
                search_result_context.add_span_to_context(
                    hit.file_path, span.span_id, add_extra=False
                )

        search_result_str = search_result_context.create_prompt(
            show_span_ids=False,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            # only_signatures=True
        )

        prompt = self._select_span_instructions(search_result)
        prompt += f"\n<search_results>\n{search_result_str}\n</search_result>\n"
        return prompt

    def _search(self, args: SearchBaseArgs) -> SearchCodeResponse:
        raise NotImplementedError("Subclasses must implement this method.")

    def _search_for_alternative_suggestion(
        self, args: SearchBaseArgs
    ) -> SearchCodeResponse:
        return SearchCodeResponse()

    def _identify_code(
        self, args: SearchBaseArgs, search_result_ctx: FileContext
    ) -> Tuple[IdentifiedSpans, Completion]:
        search_result_str = search_result_ctx.create_prompt(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
        )

        content = "Search request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections in the search results to use them. "
        content += f"\n\n<search_results>\n{search_result_str}\n</search_result>\n"
        identify_message = UserMessage(content=content)

        messages = [identify_message]
        completion = None

        MAX_RETRIES = 3
        for retry_attempt in range(MAX_RETRIES):
            identified_code, completion = self.completion_model.create_completion(
                messages=messages,
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                response_model=Identify,
            )
            logger.info(
                f"Identifying relevant code sections. Attempt {retry_attempt + 1} of {MAX_RETRIES}.\n{identified_code.identified_spans}"
            )

            view_context = FileContext(repo=self._repository)
            if identified_code.identified_spans:
                for identified_spans in identified_code.identified_spans:
                    view_context.add_line_span_to_context(
                        identified_spans.file_path,
                        identified_spans.start_line,
                        identified_spans.end_line,
                        add_extra=True,
                    )
            else:
                return view_context, completion

            tokens = view_context.context_size()

            if tokens > self.max_identify_tokens:
                logger.info(
                    f"Identified code sections are too large ({tokens} tokens)."
                )

                messages.append(
                    AssistantMessage(content=identified_code.model_dump_json())
                )

                messages.append(
                    UserMessage(
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_search_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections."
                    )
                )
            else:
                logger.info(
                    f"Identified code sections are within the token limit ({tokens} tokens)."
                )
                return view_context, completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {self.max_search_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
        )

    @classmethod
    def model_validate(cls, obj: Any) -> "SearchBaseAction":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository = obj.pop("repository")
            code_index = obj.pop("code_index")
            return cls(code_index=code_index, repository=repository, **obj)
        return super().model_validate(obj)
