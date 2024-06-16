import fnmatch
import logging
from typing import Optional, Type

from pydantic import BaseModel, Field

from moatless.file_context import FileContext, RankedFileSpan
from moatless.state import InitialState, ActionResponse
from moatless.types import (
    ActionRequest,
    Message,
    UserMessage,
    AssistantMessage,
)

logger = logging.getLogger(__name__)


SEARCH_SYSTEM_PROMPT = """You are an autonomous AI assistant.
Your task is to locate the code relevant to a users instructions using the search query action.

# Instructions:

1. Understand User Instructions:
Read the <instructions> tag to understand the specific requirements from the user.

2. Review Current File Context:
Examine the <file_context> tag to see which files and code spans have already been identified.

3. Consider the Necessary Search Parameters:
Determine if specific file types, directories, function or class names or code patterns are mentioned in the instructions.
If you can you should always try to specify the search parameters as accurately as possible.

4. Ensure At Least One Search Parameter:
Make sure that at least one of query, code_snippet, class_name, or function_name is provided.

5. Formulate the Search function:
Set at least one of the search paramaters `query`, `code_snippet`, `class_name` or `function_name`.

6. Execute the Search function:
Use the Search function with the search parameters and your thoughts on how to approach this task.

Think step by step and write out your thoughts in the thoughts field.
"""


SEARCH_FUNCTIONS_FEW_SHOT_OPENAI_FUNC = """Examples:

User:
The file uploader intermittently fails with "TypeError: cannot unpack non-iterable NoneType object". This issue appears sporadically during high load conditions..

AI Assistant:
functions.Search({
    query: "File upload process to fix intermittent 'TypeError: cannot unpack non-iterable NoneType object'",
    file_pattern: "**/uploader/**/*.py"
)

User:
There's a bug in the PaymentProcessor class where transactions sometimes fail to log correctly, resulting in missing transaction records.

AI Assistant:
functions.Search({
    class_name: "PaymentProcessor"
)

User:
The generate_report function sometimes produces incomplete reports under certain conditions. This function is part of the reporting module. Locate the generate_report function in the reports directory to debug and fix the issue.

AI Assistant:
functions.Search({
    function_name: "generate_report",
    file_pattern: "**/reports/**/*.py"
)

User:
The extract_data function in HTMLParser throws an "AttributeError: 'NoneType' object has no attribute 'find'" error when parsing certain HTML pages.

AI Assistant:
functions.Search({
    class_name: "HTMLParser",
    function_name: "extract_data"
)

User:
The database connection setup is missing SSL configuration, causing insecure connections.

Here’s the stack trace of the error:

File "/opt/app/db_config/database.py", line 45, in setup_connection
    engine = create_engine(DATABASE_URL)
File "/opt/app/db_config/database.py", line 50, in <module>
    connection = setup_connection()

AI Assistant:
functions.Search({
    code_snippet: "engine = create_engine(DATABASE_URL)",
    file_pattern: "db_config/database.py"
)
"""

SEARCH_FUNCTIONS_FEW_SHOT = """Examples:

User:
The file uploader intermittently fails with "TypeError: cannot unpack non-iterable NoneType object". This issue appears sporadically during high load conditions..

Search parameters:
    query: "File upload process to fix intermittent 'TypeError: cannot unpack non-iterable NoneType object'",
    file_pattern: "**/uploader/**/*.py"


User:
There's a bug in the PaymentProcessor class where transactions sometimes fail to log correctly, resulting in missing transaction records.

Search parameters:
    class_name: "PaymentProcessor"


User:
The generate_report function sometimes produces incomplete reports under certain conditions. This function is part of the reporting module. Locate the generate_report function in the reports directory to debug and fix the issue.

Search parameters:
    function_name: "generate_report",
    file_pattern: "**/reports/**/*.py"


User:
The extract_data function in HTMLParser throws an "AttributeError: 'NoneType' object has no attribute 'find'" error when parsing certain HTML pages.

Search parameters:
    class_name: "HTMLParser",
    function_name: "extract_data"


User:
The database connection setup is missing SSL configuration, causing insecure connections.

Here’s the stack trace of the error:

File "/opt/app/db_config/database.py", line 45, in setup_connection
    engine = create_engine(DATABASE_URL)
File "/opt/app/db_config/database.py", line 50, in <module>
    connection = setup_connection()

Search parameters:
    code_snippet: "engine = create_engine(DATABASE_URL)",
    file_pattern: "db_config/database.py"

"""

IGNORE_TEST_PROMPT = (
    "Test files are not in the search scope. Ignore requests to search for tests. "
)


class Search(ActionRequest):
    """Take action to search for code, identify found and finish up."""

    thoughts: str = Field(description="Your thoughts on what search parameters to set.")

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

    def has_search_attributes(self):
        return any(
            [
                self.query,
                self.code_snippet,
                self.class_name,
                self.function_name,
            ]
        )


class ActionCallWithContext(BaseModel):
    action: ActionRequest
    file_context: FileContext
    message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class SearchCode(InitialState):

    message: Optional[str] = Field(
        None,
        description="Message to the search",
    )

    support_test_files: bool = False

    def __init__(self, message: Optional[str] = None, **data):
        super().__init__(message=message, include_message_history=True, **data)

    def handle_action(self, action: Search) -> ActionResponse:
        if not action.has_search_attributes():
            return ActionResponse.retry(
                "You must provide at least one the search attributes query, code_snippet, class_name or function_name to search. If you're finished, set finished to true."
            )

        if (
            not self.support_test_files
            and action.file_pattern
            and is_test_pattern(action.file_pattern)
        ):
            return ActionResponse.retry("It's not possible to search for test files.")

        search_result = self.workspace.code_index.search(
            file_pattern=action.file_pattern,
            query=action.query,
            code_snippet=action.code_snippet,
            class_name=action.class_name,
            function_name=action.function_name,
        )

        logger.info(f"Found {len(search_result.hits)} hits.")

        if not search_result.hits:
            return ActionResponse.retry(
                "No code found matching the search parameters. Please try again with different search parameters."
            )

        ranked_spans = []
        for hit in search_result.hits:
            for span in hit.spans:
                ranked_spans.append(
                    RankedFileSpan(
                        file_path=hit.file_path,
                        span_id=span.span_id,
                        rank=span.rank,
                    )
                )

        output = {"ranked_spans": ranked_spans}
        output.update(action.dict(exclude={"thoughts"}))

        return ActionResponse.transition(
            trigger="did_search",
            output=output,
        )

    def action_type(self) -> Optional[Type[BaseModel]]:
        return Search

    def system_prompt(self) -> str:
        system_prompt = SEARCH_SYSTEM_PROMPT + SEARCH_FUNCTIONS_FEW_SHOT
        if not self.support_test_files:
            system_prompt += IGNORE_TEST_PROMPT
        return system_prompt

    def messages(self) -> list[Message]:
        messages: list[Message] = []

        content = (
            f"<instructions>\n{self.loop.trajectory.initial_message}\n</instructions>"
        )

        previous_transitions = self.loop.trajectory.get_transitions(str(self))
        for transition in previous_transitions:
            if transition.state.message:
                content += transition.state.message
            messages.append(UserMessage(content=content))
            messages.append(
                AssistantMessage(
                    action=transition.actions[-1].action,
                )
            )
            content = ""

        if self.file_context.files:
            file_context_str = self.file_context.create_prompt(
                exclude_comments=True,
                show_outcommented_code=True,
                outcomment_code_comment="... rest of the code",
            )
        else:
            file_context_str = "No files found yet."

        content += f"\n\n<file_context>\n{file_context_str}\n</file_context>"

        messages.append(UserMessage(content=content))
        messages.extend(self.retry_messages())

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
