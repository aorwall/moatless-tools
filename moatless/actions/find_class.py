import logging
from typing import ClassVar

from moatless.actions.schema import ActionArguments
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.completion.schema import FewShotExample
from moatless.index.types import SearchCodeResponse
from pydantic import ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class FindClassArgs(SearchBaseArgs):
    """Use this when you know the exact name of a class you want to find.

    Perfect for:
    - Finding class implementations: class_name="UserRepository"
    - Locating test classes: class_name="TestUserAuthentication"
    - Finding base classes: class_name="BaseController"
    - Finding classes in specific modules: class_name="Config", file_pattern="src/config/*.py"
    """

    class_name: str = Field(
        ...,
        description="Specific class name to search for (e.g., 'UserRepository', not 'app.models.UserRepository').",
    )

    @model_validator(mode="after")
    def validate_names(self):
        if not self.class_name.strip():
            raise ValueError("class_name cannot be empty")
        # Extract just the class name if a fully qualified name is provided
        if "." in self.class_name:
            original_name = self.class_name
            self.class_name = self.class_name.split(".")[-1]
            logger.info(f"Using class name '{self.class_name}' from fully qualified name '{original_name}'")
        return self

    model_config = ConfigDict(title="FindClass")

    def to_prompt(self):
        prompt = f"Searching for class: {self.class_name}"
        if self.file_pattern:
            prompt += f" in files matching the pattern: {self.file_pattern}"
        return prompt

    def short_summary(self) -> str:
        param_str = f"class_name={self.class_name}"
        if self.file_pattern:
            param_str += f", file_pattern={self.file_pattern}"
        return f"{self.name}({param_str})"

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="I need to see the implementation of the DatabaseManager class to understand how it handles transactions",
                action=FindClassArgs(
                    thoughts="To examine how the DatabaseManager class handles transactions, we need to locate its implementation in the codebase.",
                    class_name="DatabaseManager",
                ),
            ),
            FewShotExample.create(
                user_input="Show me the UserAuthentication class in the auth module",
                action=FindClassArgs(
                    thoughts="Looking for the UserAuthentication class specifically in the authentication module.",
                    class_name="UserAuthentication",
                    file_pattern="auth/*.py",
                ),
            ),
        ]


class FindClass(SearchBaseAction):
    args_schema: ClassVar[type[ActionArguments]] = FindClassArgs

    async def _search(self, args: FindClassArgs) -> SearchCodeResponse:
        logger.info(f"{self.name}: {args.class_name} (file_pattern: {args.file_pattern})")
        return await self.code_index.find_class(args.class_name, file_pattern=args.file_pattern)

    def _select_span_instructions(self, search_result: SearchCodeResponse) -> str:
        return (
            "Here's the class structure."
            "Use the function ViewCode and specify the SpanIDs of the relevant functions to view them.\n"
        )

    async def _search_for_alternative_suggestion(self, args: FindClassArgs) -> SearchCodeResponse:
        if args.file_pattern:
            return await self.code_index.find_class(args.class_name, file_pattern=None)
        return SearchCodeResponse()

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> list[str]:
        criteria = super().get_evaluation_criteria(trajectory_length)
        criteria.extend(
            [
                "Identifier Correctness: Verify that the class name is accurate.",
            ]
        )
        return criteria
