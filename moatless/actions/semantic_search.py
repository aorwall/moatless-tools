from typing import Optional, List, Type, ClassVar

from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments, FewShotExample
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.index.types import SearchCodeResponse


class SemanticSearchArgs(SearchBaseArgs):
    """Use this when you don't know exact names or code but want to find related functionality.

    Perfect for:
    - Finding functionality by description: query="code that handles password hashing"
    - Finding related test cases: query="tests for user registration", category="test"
    - Finding implementations: query="database connection pooling", category="implementation"
    - Finding patterns: query="error handling for API requests"

    This is the most flexible search when you:
    - Don't know exact function/class names
    - Want to find similar implementations
    - Need to discover related code
    - Want to explore how certain features are implemented
    """

    query: str = Field(
        ..., description="Natural language description of what you're looking for."
    )
    category: Optional[str] = Field(
        None,
        description="The category of files to search for. This can be 'implementation' for core implementation files or 'test' for test files.",
    )

    class Config:
        title = "SemanticSearch"

    def to_prompt(self):
        prompt = f"Searching for code using the query: {self.query}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    @model_validator(mode="after")
    def validate_query(self) -> "SemanticSearchArgs":
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        return self


class SemanticSearch(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = SemanticSearchArgs

    def _search(self, args: SemanticSearchArgs) -> SearchCodeResponse:
        return self._code_index.semantic_search(
            args.query,
            file_pattern=args.file_pattern,
            max_results=self.max_hits,
            category=args.category,
        )

    def _search_for_alternative_suggestion(
        self, args: SemanticSearchArgs
    ) -> SearchCodeResponse:
        if args.file_pattern:
            return self._code_index.semantic_search(
                args.query,
                max_results=self.max_hits,
                category=args.category,
            )

        return SearchCodeResponse()

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> List[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Query Relevance: Evaluate if the search query is well-defined and likely to find relevant code.",
                "Category Appropriateness: Assess if the category (implementation or test) aligns with the search intent.",
            ]
        )
        return criteria

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Find all implementations of database connection pooling in our codebase",
                action=SemanticSearchArgs(
                    scratch_pad="To find implementations of database connection pooling, we should search for code related to managing database connections efficiently. This might include classes or functions that handle connection creation, reuse, and management.",
                    query="database connection pooling implementation",
                    category="implementation",
                ),
            ),
            FewShotExample.create(
                user_input="We need to find all test cases related to user authentication in our test suite",
                action=SemanticSearchArgs(
                    scratch_pad="To find test cases related to user authentication, we should search for test files that contain assertions and scenarios specifically testing authentication functionality.",
                    query="user authentication test cases",
                    file_pattern="tests/*.py",
                    category="test",
                ),
            ),
        ]
