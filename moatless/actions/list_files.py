from typing import List

from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.model import (
    ActionArguments,
    Observation,
    FewShotExample,
)
from moatless.file_context import FileContext
from moatless.workspace import Workspace


class ListFilesArgs(ActionArguments):
    """List files and directories in a specified directory."""

    directory: str = Field(
        default="",
        description="The directory path to list. Empty string means root directory.",
    )

    class Config:
        title = "ListFiles"

    def to_prompt(self):
        return f"List contents of directory: {self.directory or '(root)'}"

    def short_summary(self) -> str:
        param_str = f"directory={self.directory}"
        return f"{self.name}({param_str})"


class ListFiles(Action):
    args_schema = ListFilesArgs

    def execute(
        self,
        args: ListFilesArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if not file_context._repo:
            raise RuntimeError("Repository not available for listing files.")

        try:
            result = file_context._repo.list_directory(args.directory)

            message = f"Contents of directory '{args.directory or '(root)'}'\n\n"

            if result["directories"]:
                message += "Directories:\n"
                for directory in result["directories"]:
                    message += f"📁 {directory}\n"
                message += "\n"

            if result["files"]:
                message += "Files:\n"
                for file in result["files"]:
                    message += f"📄 {file}\n"

            if not result["directories"] and not result["files"]:
                message += "Directory is empty or does not exist."

            return Observation(
                message=message,
                summary=message,  # f"Listed contents of directory '{args.directory or '(root)'}'",
                properties=result,
                expect_correction=False,
            )

        except Exception as e:
            return Observation(
                message=f"Error listing directory: {str(e)}",
                expect_correction=True,
            )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length) -> List[str]:
        return [
            "Directory Path Validity: Ensure the requested directory path exists and is valid.",
            "Usefulness: Assess if listing the directory contents is helpful for the current task.",
            "Efficiency: Evaluate if the action is being used at an appropriate time in the workflow.",
        ]

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Show me what files are in the tests directory",
                action=ListFilesArgs(
                    thoughts="I'll list the contents of the tests directory to see what test files are available.",
                    directory="tests",
                ),
            ),
            FewShotExample.create(
                user_input="What files are in the root directory?",
                action=ListFilesArgs(
                    thoughts="I'll list the contents of the root directory to see the project structure.",
                    directory="",
                ),
            ),
        ]
