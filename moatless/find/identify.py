import fnmatch
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from moatless.codeblocks import CodeBlockType
from moatless.file_context import FileContext, RankedFileSpan
from moatless.state import AgenticState
from moatless.types import (
    FileWithSpans,
    ActionRequest,
    ActionResponse,
    Message,
    UserMessage,
)

logger = logging.getLogger(__name__)


IDENTIFY_SYSTEM_PROMPT = """You are an autonomous AI assistant tasked with finding relevant code in an existing 
codebase based on user instructions. Your task is to identify the relevant code spans in the provided search 
results and decide whether the search task is complete.

# Input Structure:

* <instructions>: Contains the user's instructions for identifying relevant code.
* <file_context>: Contains the context of already identified files and code spans.
* <search_query>: Contains the search query used to obtain new results.
* <search_results>: Contains the new search results with code divided into "code spans".

# Your Task:

1. Analyze User Instructions:
Carefully read the user's instructions within the <instructions> tag.

2. Review Current Context:
Examine the current file context provided in the <file_context> tag to understand already identified relevant files.

3. Process New Search Results:
Analyze the new search results within the <search_results> tag.
Identify and extract relevant code spans based on the user's instructions.

4. Make a Decision:
* If you believe all relevant files are identified, mark the task as complete.
* If you believe more relevant files can be identified, mark the task as not complete to continue the search.

5. Respond Using the Function:
Use the Identify function to provide your response.

Think step by step and write out your thoughts in the thoughts field.
"""


class Identify(ActionRequest):
    """Identify if the provided search result is relevant to the users instructions."""

    thoughts: str = Field(
        description="Your thoughts on if the spans where relevant or not and if you found all relevant spans and can finish.."
    )

    identified_spans: Optional[List[FileWithSpans]] = Field(
        default=None,
        description="Files and code spans in the search results identified as relevant to the users instructions.",
    )

    complete: bool = Field(
        default=False,
        description="Set to true if all the relevant code spans have been identified.",
    )


class IdentifyCode(AgenticState):

    file_pattern: Optional[str]
    query: Optional[str]
    code_snippet: Optional[str]
    class_name: Optional[str]
    function_name: Optional[str]
    ranked_spans: Optional[List[RankedFileSpan]]

    def __init__(
        self,
        file_pattern: str,
        query: str,
        code_snippet: str,
        class_name: str,
        function_name: str,
        ranked_spans: List[RankedFileSpan],
        **data,
    ):
        super().__init__(
            file_pattern=file_pattern,
            query=query,
            code_snippet=code_snippet,
            class_name=class_name,
            function_name=function_name,
            ranked_spans=ranked_spans,
            include_message_history=False,
            **data,
        )

    def handle_action(self, action: Identify) -> ActionResponse:
        if action.identified_spans:
            self.file_context.add_files_with_spans(action.identified_spans)

            span_count = sum([len(file.span_ids) for file in action.identified_spans])
            logger.info(
                f"Identified {span_count} spans in {len(action.identified_spans)} files. Current file context size is {self.file_context.context_size()} tokens."
            )
        else:
            logger.info("No spans identified.")

        if not self.ranked_spans:
            message = "The search did not return any code spans."
        else:
            message = f"The search returned {len(self.ranked_spans)} code spans. "

        if action.identified_spans:
            message += "\n\nIdentified the following code spans in the search result to be relevant:"
            for file in action.identified_spans:
                span_str = ", ".join(file.span_ids)
                message += f"\n * {file.file_path}: {(span_str)}:"

        else:
            message += (
                "\n\nNo code spans in the search result was identified as relevant."
            )

        message += "\n\n"
        message += action.thoughts

        if action.complete:
            return ActionResponse.transition(
                "finish",
                output={"message": action.thoughts},
            )
        else:
            return ActionResponse.transition(
                "search",
                output={"message": message},
            )

    def action_type(self) -> Optional[Type[BaseModel]]:
        return Identify

    def system_prompt(self) -> str:
        return IDENTIFY_SYSTEM_PROMPT

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        search_query = ""
        if self.query:
            search_query = f"Query: {self.query}\n"
        if self.code_snippet:
            search_query = f"Exact code match: {self.code_snippet}\n"
        if self.class_name:
            search_query = f"Class name: {self.class_name}\n"
        if self.function_name:
            search_query = f"Function name: {self.function_name}\n"

        file_context = self.create_file_context()
        file_context.add_ranked_spans(self.ranked_spans)

        if file_context.files:
            search_result_str = file_context.create_prompt(
                show_span_ids=True,
                show_line_numbers=False,
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            search_result_str = "No new search results found."

        if self.file_context.files:
            file_context_str = self.file_context.create_prompt(
                show_span_ids=True,
                show_line_numbers=False,
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            file_context_str = "No relevant code identified yet."

        content = f"""<instructions>
{self.loop.trajectory.initial_message}
</instructions>

<file_context>
{file_context_str}
</file_context>

<search_query>
{search_query}
</search_query>

<search_results>
{search_result_str}
</search_results>
"""

        messages.append(UserMessage(content=content))
        return messages

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
