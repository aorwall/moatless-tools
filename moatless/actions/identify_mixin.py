import logging
from typing import Optional, Tuple, List

from moatless.actions.schema import ActionArguments
from pydantic import BaseModel, Field

from moatless.actions.action import CompletionModelMixin
from moatless.completion.model import Completion
from moatless.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
    ResponseSchema,
)
from moatless.exceptions import CompletionRejectError
from moatless.completion.base import CompletionRetryError
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with intelligently reducing and identifying the most relevant code sections from an oversized codebase response. Your primary goal is to select only the essential code sections that directly relate to the original search request, helping to reduce the context size while maintaining relevance.

The previous messages will contain:
1. A search request from an AI assistant that resulted in too much code being returned
2. Search results containing various code sections with their line numbers that need to be filtered

# Your Task:

1. Understand the Original Search Request:
   * Analyze the previous search request to understand what code elements are being looked for
   * Identify the core requirements and key elements that are absolutely necessary
   * Keep in mind that the goal is to reduce the amount of code while maintaining relevance

2. Evaluate and Filter Search Results:
   * Carefully examine each code section and prioritize based on direct relevance
   * Select only the most important and directly relevant code sections
   * Avoid including peripheral or contextually similar but non-essential code
   * Focus on reducing the overall size while maintaining the critical functionality

3. Respond with Minimal but Complete Selection:
   * Choose only the most relevant code sections that directly answer the search request
   * Provide your analysis in the thoughts field, explaining why certain sections were chosen and others excluded
   * List the relevant file paths with start and end line numbers in the identified_spans field
   * Ensure your selection stays within token limits while capturing the essential code"""


class IdentifiedSpans(BaseModel):
    file_path: str = Field(description="The file path where the relevant code is found.")
    start_line: int = Field(description="Starting line number of the relevant code section.")
    end_line: int = Field(description="Ending line number of the relevant code section.")


class Identify(ResponseSchema):
    """Identify if the provided search result is relevant to the reported issue."""

    thoughts: Optional[str] = Field(
        None,
        description="Your thoughts and analysis on the search results and how they relate to the reported issue.",
    )

    identified_spans: Optional[list[IdentifiedSpans]] = Field(
        default=None,
        description="Files and code sections in the search results identified as relevant to the reported issue.",
    )


class IdentifyMixin(CompletionModelMixin):
    """Mixin that provides identify flow functionality for large code sections."""

    max_identify_tokens: int = Field(
        8000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_identify_prompt_tokens: int = Field(
        16000,
        description="The maximum number of tokens allowed in the identify prompt.",
    )

    def _initialize_completion_model(self):
        """Initialize the completion model with validation function for token limits"""
        async def validate_identified_code(
            structured_outputs: List[ResponseSchema],
            text_response: Optional[str],
            flags: List[str],
        ) -> Tuple[List[ResponseSchema], Optional[str], List[str]]:
            identified_context = FileContext(repo=self._repository)
            
            if not structured_outputs:
                return structured_outputs, text_response, flags
                
            for identified_code in structured_outputs:
                if identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        identified_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

            tokens = identified_context.context_size()
            if tokens > self.max_identify_tokens:
                logger.warning(f"Identified code is too large ({tokens} tokens). Maximum allowed is {self.max_identify_tokens} tokens. ")
                raise CompletionRetryError(
                    f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_identify_tokens} tokens. "
                    f"Please identify a smaller subset of the most relevant code sections."
                )
                
            return structured_outputs, text_response, flags

        self._completion_model.initialize(Identify, IDENTIFY_SYSTEM_PROMPT, post_validation_fn=validate_identified_code)

    async def _identify_code(self, args: ActionArguments, view_context: FileContext, max_tokens: int) -> Tuple[FileContext, Completion]:
        """Identify relevant code sections in a large context.

        Args:
            args: The arguments containing the request information
            view_context: The context containing the code to identify from
            max_tokens: The maximum number of tokens allowed in the result

        Returns:
            A tuple of (identified_context, completion)
        """

        code_str = view_context.create_prompt(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            max_tokens=self.max_identify_prompt_tokens,
        )

        content = "A previous request for the following query resulted in too much code:\n"
        content += f"\nOriginal Request:\n{args.to_prompt()}\n"
        content += "\nThis search returned more code than can be processed efficiently. Your task is to analyze the results below and select only the most essential code sections that directly address the original request. The goal is to reduce the context size while preserving the most relevant code."

        content += "\n\nBelow are all the code sections that were found. Please identify and select only the most critical parts that directly answer the original request:"
        content += f"\n\n<code>\n{code_str}\n</code>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)

        messages = [identify_message]
        completion_response = await self._completion_model.create_completion(messages=messages)

        identified_context = FileContext(repo=self._repository)
        if completion_response.structured_outputs:
            for identified_code in completion_response.structured_outputs:
                if identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        identified_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

        return identified_context, completion_response.completion
