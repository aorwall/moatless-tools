import logging
from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from moatless.actions.action import CompletionModelMixin
from moatless.completion import BaseCompletionModel
from moatless.completion.model import Completion
from moatless.completion.schema import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
    ResponseSchema,
)
from moatless.exceptions import CompletionRejectError
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

    def _identify_code(self, args, view_context: FileContext, max_tokens: int) -> Tuple[FileContext, Completion]:
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

        content = "Code request:"
        content += f"\n{args.to_prompt()}"

        content += "\n\nIdentify the relevant code sections to view. "
        content += f"\n\n<code>\n{code_str}\n</code>\n"
        identify_message = ChatCompletionUserMessage(role="user", content=content)

        messages = [identify_message]
        completion = None

        MAX_RETRIES = 3
        for retry_attempt in range(MAX_RETRIES):
            completion_response = self._completion_model.create_completion(messages=messages)
            logger.info(
                f"Identifying relevant code sections. Attempt {retry_attempt + 1} of {MAX_RETRIES}.{len(completion_response.structured_outputs)} identify requests."
            )

            identified_context = FileContext(repo=self._repository)
            if not completion_response.structured_outputs:
                logger.warning("No identified code in response")
                return identified_context, completion_response.completion

            for identified_code in completion_response.structured_outputs:
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
                logger.info(f"Identified code sections are too large ({tokens} tokens).")

                messages.append(
                    ChatCompletionAssistantMessage(role="assistant", content=identified_code.model_dump_json())
                )

                messages.append(
                    ChatCompletionUserMessage(
                        role="user",
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {max_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections.",
                    )
                )
            else:
                logger.info(f"Identified code sections are within the token limit ({tokens} tokens).")
                return identified_context, completion_response.completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {max_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
            messages=messages,
        )

    def _initialize_completion_model(self):
        self._completion_model.initialize(Identify, IDENTIFY_SYSTEM_PROMPT)
