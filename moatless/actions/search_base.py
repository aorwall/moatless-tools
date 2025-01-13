import logging
from abc import ABC
from typing import List, Optional, Type, Any, ClassVar, Tuple

from litellm.types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from pydantic import Field, PrivateAttr, BaseModel, field_validator

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, Observation, RewardScaleEntry
from moatless.completion import CompletionModel
from moatless.completion.model import Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse
from moatless.repository.repository import Repository
from moatless.workspace import Workspace

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
   * Provide your analysis in the thoughts field
   * List the relevant file paths with start and end line numbers in the identified_spans field
"""


class SearchBaseArgs(ActionArguments, ABC):
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific files or directories.",
    )

    @field_validator("file_pattern")
    @classmethod
    def validate_file_pattern(cls, v):
        if v:
            if "," in v:
                raise ValueError("File pattern cannot contain commas")
        return v


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


class Identify(StructuredOutput):
    """Identify if the provided search result is relevant to the reported issue."""

    thoughts: Optional[str] = Field(
        None,
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
    max_identify_prompt_tokens: int = Field(
        16000,
        description="The maximum number of tokens allowed in the identify prompt.",
    )
    max_hits: int = Field(
        10,
        description="The maximum number of search hits to display.",
    )
    completion_model: CompletionModel = Field(
        ...,
        description="The completion model used to identify relevant code sections in search results.",
    )

    _repository: Repository = PrivateAttr()
    _code_index: CodeIndex = PrivateAttr()

    def __init__(
        self,
        repository: Repository = None,
        code_index: CodeIndex | None = None,
        completion_model: CompletionModel = None,
        **data,
    ):
        super().__init__(completion_model=completion_model, **data)
        self._repository = repository
        self._code_index = code_index

    def execute(
        self,
        args: SearchBaseArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if file_context is None:
            raise ValueError(
                "File context must be provided to execute the search action."
            )

        properties = {"search_hits": [], "search_tokens": 0}

        search_result_context, alternative_suggestion = self._search_for_context(args)

        if search_result_context.is_empty():
            properties["fail_reason"] = "no_search_hits"
            return Observation(message="No search results found", properties=properties)

        properties["search_tokens"] = search_result_context.context_size()
        properties["search_hits"] = search_result_context.model_dump(exclude_none=True)

        completion = None

        if (
            search_result_context.span_count() == 1
            and search_result_context.context_size() > self.max_identify_tokens
        ):
            logger.warning(
                f"{self.name}: Conext for {search_result_context.create_summary()} is too large ({search_result_context.context_size()} tokens)."
            )
            properties["fail_reason"] = "search_too_large"
            return Observation(
                message="Search too large. Found a single code section that is too large to view. Please refine the search query.",
                properties=properties,
            )
        elif (
            search_result_context.context_size() > self.max_search_tokens
            and search_result_context.span_count() > 1
        ) or search_result_context.span_count() > self.max_hits:
            logger.info(
                f"{self.name}: Search too large. {properties['search_tokens']} tokens and {search_result_context.span_count()} hits, will ask for clarification."
            )
            view_context, completion = self._identify_code(args, search_result_context)
        else:
            view_context = search_result_context

        span_count = search_result_context.span_count()
        search_result_str = f"Found {span_count} code sections."

        # Apply previous changes to view context
        # TODO: Refactor
        for file in file_context.files:
            if view_context.has_file(file.file_path) and file.patch:
                view_context.get_file(file.file_path).set_patch(file.patch)

        new_span_ids = file_context.add_file_context(view_context)

        if view_context.is_empty():
            search_result_str += (
                "\n\nNone of the search results was relevant to the task."
            )
            summary = "Didn't find any relevant code sections in the search results."
            message = search_result_str
        else:
            viewed_str = "that has already been viewed" if not new_span_ids else ""

            if alternative_suggestion:
                summary = f"Did not find an exact match but found the following alternative suggestions {viewed_str}:\n{view_context.create_summary()}"
            else:
                summary = f"Found the following relevant code spans {viewed_str}:\n{view_context.create_summary()}"

            message = "Found the following relevant code:\n"
            message += view_context.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                exclude_comments=False,
                show_outcommented_code=True,
            )

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

    def _search_for_context(self, args: SearchBaseArgs) -> Tuple[FileContext, bool]:
        alternative_suggestion = False
        search_result = self._search(args)
        if not search_result.hits:
            search_result = self._search_for_alternative_suggestion(args)
            alternative_suggestion = True
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

        return search_result_context, alternative_suggestion

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
            max_tokens=self.max_identify_prompt_tokens,
        )

        content = "Search request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections in the search results to use them. "
        content += f"\n\n<search_results>\n{search_result_str}\n</search_result>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)

        messages = [identify_message]
        completion = None

        MAX_RETRIES = 3
        for retry_attempt in range(MAX_RETRIES):
            completion_response = self.completion_model.create_completion(
                messages=messages,
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                response_model=Identify,
            )
            logger.info(
                f"Identifying relevant code sections. Attempt {retry_attempt + 1} of {MAX_RETRIES}.{len(completion_response.structured_outputs)} identify requests."
            )

            view_context = FileContext(repo=self._repository)
            if not completion_response.structured_outputs:
                logger.warning("No identified code in response")
                return view_context, completion_response.completion

            for identified_code in completion_response.structured_outputs:
                if identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        view_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

            tokens = view_context.context_size()

            if tokens > self.max_identify_tokens:
                logger.info(
                    f"Identified code sections are too large ({tokens} tokens)."
                )

                messages.append(
                    ChatCompletionAssistantMessage(
                        role="assistant", content=identified_code.model_dump_json()
                    )
                )

                messages.append(
                    ChatCompletionUserMessage(
                        role="user",
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_search_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections.",
                    )
                )
            else:
                logger.info(
                    f"Identified code sections are within the token limit ({tokens} tokens)."
                )
                return view_context, completion_response.completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {self.max_search_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
        )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        evaluation_criteria = super().get_evaluation_criteria(trajectory_length)
        evaluation_criteria.extend(
            [
                "Query Relevance: Evaluate if the search query or parameters are well-defined and likely to find relevant code.",
                "Search Scope Appropriateness: Check if the file patterns and class/function names narrow down the search effectively.",
                "Relevance of Search Results: Assess whether the search results are directly related to the problem and useful for making progress.",
                "Size of Search Results: Ensure that the code context provided is appropriately sized—not too large to overwhelm nor too small to be unhelpful.",
            ]
        )

        return evaluation_criteria

    @classmethod
    def get_reward_scale(cls, trajectory_length) -> List[RewardScaleEntry]:
        if trajectory_length <= 3:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action is excellent, with well-defined parameters yielding only highly relevant results.",
                    ),
                    (
                        75,
                        89,
                        "The search action is good, with reasonable parameters yielding relevant results.",
                    ),
                    (
                        25,
                        74,
                        "The search action have issues with parameters or yields few or no relevant results.",
                    ),
                    (
                        0,
                        24,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )
        else:
            return cls.generate_reward_scale_entries(
                [
                    (
                        90,
                        100,
                        "The search action significantly advances the solution, providing highly relevant and appropriately sized search results.",
                    ),
                    (
                        75,
                        89,
                        "The search action contributes positively towards solving the problem, with relevant results and minor issues.",
                    ),
                    (
                        50,
                        74,
                        "The search action is acceptable but may have issues with relevance or provides search results that are too large or too small.",
                    ),
                    (
                        25,
                        49,
                        "The search action provides results that are not helpful due to relevance or size issues.",
                    ),
                    (
                        0,
                        24,
                        "The search action has minimal impact, providing few relevant results.",
                    ),
                    (
                        -50,
                        -1,
                        "The action is counterproductive, with search results that are entirely irrelevant or excessively large, causing setbacks.",
                    ),
                ]
            )

    @classmethod
    def model_validate(cls, obj: Any) -> "SearchBaseAction":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository = obj.pop("repository")
            code_index = obj.pop("code_index")
            return cls(code_index=code_index, repository=repository, **obj)
        return super().model_validate(obj)
