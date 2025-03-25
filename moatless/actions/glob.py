from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments
from moatless.file_context import FileContext
from pydantic import ConfigDict, Field


class GlobArgs(ActionArguments):
    """
    - Fast file pattern matching tool that works with any codebase size
    - Supports glob patterns like "**/*.js" or "src/**/*.ts"
    - Use this tool when you need to find files by name patterns
    """

    pattern: str = Field(
        ...,
        description="The glob pattern to match files (e.g. '**/*.js', 'src/**/*.ts').",
    )

    max_results: int = Field(
        100,
        description="Maximum number of results to return",
    )

    model_config = ConfigDict(title="GlobTool")

    def to_prompt(self):
        return f"Find files matching glob pattern '{self.pattern}'"

    def short_summary(self) -> str:
        param_str = f"pattern='{self.pattern}'"
        if self.max_results != 100:
            param_str += f", max_results={self.max_results}"
        return f"{self.name}({param_str})"


class GlobTool(Action):
    args_schema = GlobArgs

    async def _execute(
        self,
        args: GlobArgs,
        file_context: FileContext | None = None,
    ) -> str | None:
        if not self.workspace:
            raise ValueError("Workspace is required to run glob matching")

        if not self.workspace.repository:
            raise ValueError("Repository is required to run glob matching")

        if not hasattr(self.workspace.repository, "matching_files"):
            raise ValueError("Repository does not support glob matching")

        try:
            from moatless.repository.file import FileRepository

            repo = self.workspace.repository
            if isinstance(repo, FileRepository):
                files: list[str] = await repo.matching_files(args.pattern)
            else:
                # Fallback for other repository types that might implement matching_files
                files: list[str] = await getattr(repo, "matching_files")(args.pattern)

            if args.max_results and len(files) > args.max_results:
                files = files[: args.max_results]

            if not files or (len(files) == 1 and not files[0]):
                return f"No files found matching glob pattern '{args.pattern}'"

            message = f"Found {len(files)} files matching glob pattern '{args.pattern}':\n\n"

            for file_path in files:
                message += f"📄 {file_path}\n"

            return message

        except Exception as e:
            return f"Error executing glob search: {str(e)}"
