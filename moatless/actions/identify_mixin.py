import logging
from typing import Optional, Tuple, List

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
                        await identified_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

            tokens = await identified_context.context_size()
            if tokens > self.max_identify_tokens:
                raise CompletionRetryError(
                    f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {self.max_identify_tokens} tokens. "
                    f"Please identify a smaller subset of the most relevant code sections."
                )
                
            return structured_outputs, text_response, flags

        self._completion_model.initialize(Identify, IDENTIFY_SYSTEM_PROMPT, post_validation_fn=validate_identified_code)

    async def _identify_code(self, args, view_context: FileContext, max_tokens: int) -> Tuple[FileContext, Completion]:
        """Identify relevant code sections in a large context.

        Args:
            args: The arguments containing the request information
            view_context: The context containing the code to identify from
            max_tokens: The maximum number of tokens allowed in the result

        Returns:
            A tuple of (identified_context, completion)
        """
        code_str = await view_context.create_prompt_async(
            show_span_ids=True,
            show_line_numbers=True,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="...",
            max_tokens=self.max_identify_prompt_tokens,
        )

        content = "Code request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections to view. "
        content += f"\n\n<code>\n{code_str}\n</code>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)

        messages = [identify_message]
        completion_response = await self._completion_model.create_completion(messages=messages)

        identified_context = FileContext(repo=self._repository)
        if completion_response.structured_outputs:
            for identified_code in completion_response.structured_outputs:
                if identified_code.identified_spans:
                    for identified_spans in identified_code.identified_spans:
                        await identified_context.add_line_span_to_context(
                            identified_spans.file_path,
                            identified_spans.start_line,
                            identified_spans.end_line,
                            add_extra=True,
                        )

        return identified_context, completion_response.completion
