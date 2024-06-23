import fnmatch
import logging
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from moatless.file_context import RankedFileSpan
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
codebase based on a reported issue. Your task is to identify the relevant code spans in the provided search 
results and decide whether the search task is complete.

# Input Structure:

* <issue>: Contains the reported issue.
* <file_context>: Contains the context of already identified files and code spans.
* <search_results>: Contains the new search results with code divided into "code spans".

# Your Task:

1. Analyze User Instructions:
Carefully read the reported issue within the <issue> tag.

2. Review Current Context:
Examine the current file context provided in the <file_context> tag to understand already identified relevant files.

3. Process New Search Results:
3.1. Thoroughly analyze each code span in the <search_results> tag.
3.2. Match the code spans with the key elements, functions, variables, or patterns identified in the reported issue.
3.3. Evaluate the relevance of each code span based on how well it aligns with the reported issue and current file context.
3.4. If the issue suggests new functions or classes, identify the existing code that might be relevant to be able to implement the new functionality.
3.5. Review entire sections of code, not just isolated spans, to ensure you have a complete understanding before making a decision. It's crucial to see all code in a section to accurately determine relevance and completeness.
3.6. Verify if there are references to other parts of the codebase that might be relevant but not found in the search results. 
3.7. Identify and extract relevant code spans based on the reported issue. 

4. Respond Using the Function:
Use the Identify function to provide your response.

Think step by step and write out your thoughts in the scratch_pad field.
"""


class Identify(ActionRequest):
    """Identify if the provided search result is relevant to the reported issue."""

    scratch_pad: str = Field(
        description="Your thoughts on how to identify the relevant code and why."
    )

    identified_spans: Optional[List[FileWithSpans]] = Field(
        default=None,
        description="Files and code spans in the search results identified as relevant to the reported issue.",
    )


class IdentifyCode(AgenticState):

    file_pattern: Optional[str]
    query: Optional[str]
    code_snippet: Optional[str]
    class_name: Optional[str]
    function_name: Optional[str]

    ranked_spans: Optional[List[RankedFileSpan]]

    expand_context: bool
    max_prompt_file_tokens: int = 4000

    def __init__(
        self,
        ranked_spans: List[RankedFileSpan],
        file_pattern: Optional[str] = None,
        query: Optional[str] = None,
        code_snippet: Optional[str] = None,
        class_name: Optional[str] = None,
        function_name: Optional[str] = None,
        expand_context: bool = True,
        max_prompt_file_tokens: int = 4000,
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
            expand_context=expand_context,
            max_prompt_file_tokens=max_prompt_file_tokens,
            **data,
        )

    def handle_action(self, action: Identify) -> ActionResponse:
        if action.identified_spans:
            self.file_context.add_files_with_spans(action.identified_spans)

            span_count = sum([len(file.span_ids) for file in action.identified_spans])
            logger.info(
                f"Identified {span_count} spans in {len(action.identified_spans)} files. Current file context size is {self.file_context.context_size()} tokens."
            )

            return ActionResponse.transition("finish")
        else:
            logger.info("No spans identified.")

        message = "I searched using the following parameters:\n"

        if self.file_pattern:
            message += f"\n* **File Pattern:** `{self.file_pattern}`"
        if self.query:
            message += f"\n* **Query:** `{self.query}`"
        if self.code_snippet:
            message += f"\n* **Code Snippet:** `{self.code_snippet}`"
        if self.class_name:
            message += f"\n* **Class Name:** `{self.class_name}`"
        if self.function_name:
            message += f"\n* **Function Name:** `{self.function_name}`"

        message = f"The search returned {len(self.ranked_spans)} results. But unfortunately, I didn’t find any of the search results relevant to the query."

        message += "\n\n"
        message += action.scratch_pad

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

        file_context = self.create_file_context(max_tokens=self.max_prompt_file_tokens)
        file_context.add_ranked_spans(self.ranked_spans)

        if file_context.files:
            file_context.expand_context_with_init_spans()

            if self.expand_context:
                file_context.expand_context_with_related_spans(
                    max_tokens=self.max_prompt_file_tokens, set_tokens=True
                )
                file_context.expand_small_classes(
                    max_tokens=self.max_prompt_file_tokens
                )

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

        content = f"""<issue>
{self.loop.trajectory.initial_message}
</issue>

<file_context>
{file_context_str}
</file_context>

<search_results>
{search_result_str}
</search_results>
"""

        messages.append(UserMessage(content=content))
        return messages


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
