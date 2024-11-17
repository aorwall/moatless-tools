import logging
from typing import List, Optional, Type, ClassVar

from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments, FewShotExample
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class FindCodeSnippetArgs(SearchBaseArgs):
    """Use this when you know the exact code you want to find.
         It will run the command: grep -n -r "code_snippet" "file_pattern"

    Perfect for:
    - Finding specific constant definitions: code_snippet="MAX_RETRIES = 3"
    - Finding decorator usage: code_snippet="@retry(max_attempts=3)"
    - Finding specific imports: code_snippet="from datetime import datetime"
    - Finding configuration patterns: code_snippet="DEBUG = os.getenv('DEBUG', False)"

    Note: You must know the exact code snippet. Use SemanticSearch if you only know
    what the code does but not its exact implementation.
    """

    code_snippet: str = Field(..., description="The exact code snippet to find.")
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories. ",
    )

    class Config:
        title = "FindCodeSnippet"

    @model_validator(mode="after")
    def validate_snippet(self) -> "FindCodeSnippetArgs":
        if not self.code_snippet.strip():
            raise ValueError("code_snippet cannot be empty")
        return self

    def to_prompt(self):
        prompt = f"Searching for code snippet: {self.code_snippet}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt


class FindCodeSnippet(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = FindCodeSnippetArgs

    def _search_for_context(self, args: FindCodeSnippetArgs) -> FileContext:
        logger.info(
            f"{self.name}: {args.code_snippet} (file_pattern: {args.file_pattern})"
        )

        matches = self._repository.find_exact_matches(
            search_text=args.code_snippet, file_pattern=args.file_pattern
        )

        search_result_context = FileContext(repo=self._repository)
        for file_path, start_line in matches:
            num_lines = len(args.code_snippet.splitlines())
            end_line = start_line + num_lines - 1

            search_result_context.add_line_span_to_context(
                file_path, start_line, end_line
            )

        return search_result_context

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="I need to understand how the User class is structured in our authentication system. Let me find its definition.",
                action=FindCodeSnippetArgs(
                    scratch_pad="To find the User class definition, I'll search for the exact class declaration line 'class User(BaseModel):'",
                    code_snippet="class User(BaseModel):",
                ),
            ),
            FewShotExample.create(
                user_input="The system seems to use a default timeout value. I should check where DEFAULT_TIMEOUT is defined in the configuration.",
                action=FindCodeSnippetArgs(
                    scratch_pad="To find the timeout configuration, I'll search for the exact variable declaration 'DEFAULT_TIMEOUT =' in config files",
                    code_snippet="DEFAULT_TIMEOUT =",
                    file_pattern="**/config/*.py",
                ),
            ),
            FewShotExample.create(
                user_input="To understand how request processing works, I need to examine the _handlers dictionary in the processor service.",
                action=FindCodeSnippetArgs(
                    scratch_pad="To find the handlers mapping, I'll search for the exact dictionary declaration '_handlers =' in the processor service",
                    code_snippet="_handlers =",
                    file_pattern="services/processor.py",
                ),
            ),
        ]
