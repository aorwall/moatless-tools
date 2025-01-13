import logging
from typing import Optional, Tuple

from litellm.types.llms.openai import (
    ChatCompletionAssistantMessage,
    ChatCompletionUserMessage,
)
from pydantic import Field

from moatless.actions.search_base import IDENTIFY_SYSTEM_PROMPT, Identify
from moatless.completion import CompletionModel
from moatless.completion.model import Completion
from moatless.exceptions import CompletionRejectError
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class IdentifyMixin:
    """Mixin that provides identify flow functionality for large code sections."""

    completion_model: Optional[CompletionModel] = Field(
        None,
        description="The completion model used to identify relevant code sections.",
    )
    max_identify_tokens: int = Field(
        8000,
        description="The maximum number of tokens allowed in the identified code sections.",
    )
    max_identify_prompt_tokens: int = Field(
        16000,
        description="The maximum number of tokens allowed in the identify prompt.",
    )

    def _identify_code(
        self, args, view_context: FileContext, max_tokens: int
    ) -> Tuple[FileContext, Completion]:
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
            completion_response = self.completion_model.create_completion(
                messages=messages,
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                response_model=Identify,
            )
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
                        content=f"The identified code sections are too large ({tokens} tokens). Maximum allowed is {max_tokens} tokens. "
                        f"Please identify a smaller subset of the most relevant code sections.",
                    )
                )
            else:
                logger.info(
                    f"Identified code sections are within the token limit ({tokens} tokens)."
                )
                return identified_context, completion_response.completion

        # If we've exhausted all retries and still too large
        raise CompletionRejectError(
            f"Unable to reduce code selection to under {max_tokens} tokens after {MAX_RETRIES} attempts",
            last_completion=completion,
        )
