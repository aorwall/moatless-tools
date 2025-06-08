import logging

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


class CleanupContextArgs(ActionArguments):
    """Remove files to reduce clutter and focus on relevant files.

    This action allows you to remove files from the current file context when they are no longer
    needed for the task at hand.
    """

    file_paths: list[str] = Field(
        ...,
        description="List of file paths to remove from the file context. These should be relative to the repository root.",
        min_length=1,
    )

    model_config = ConfigDict(title="Cleanup")

    @property
    def log_name(self):
        if len(self.file_paths) == 1:
            return f"CleanupContext({self.file_paths[0]})"
        return f"CleanupContext({len(self.file_paths)} files)"

    def to_prompt(self):
        if len(self.file_paths) == 1:
            return f"Remove {self.file_paths[0]} from file context"
        return f"Remove {len(self.file_paths)} files from file context: {', '.join(self.file_paths[:3])}{'...' if len(self.file_paths) > 3 else ''}"

    def short_summary(self) -> str:
        return f"{self.name}({len(self.file_paths)} files)"


class CleanupContext(Action):
    """An action that removes files from the file context to reduce clutter.

    This action is useful for maintaining a clean and focused file context by removing
    files that are no longer relevant to the current task. It helps to:
    - Reduce context size and token usage
    - Focus on the most relevant files
    - Remove outdated or irrelevant files from consideration

    The action will:
    - Remove specified files from the file context
    - Provide feedback on which files were successfully removed
    - Report any files that were not found in the context
    - Maintain the repository files themselves (only removes from context)

    Error cases handled:
    - Files not found in the current context
    - Empty file list
    - Missing file context
    """

    args_schema = CleanupContextArgs

    async def execute(self, args: ActionArguments, file_context: FileContext | None = None) -> Observation:
        if file_context is None:
            raise ValueError("File context must be provided to execute the cleanup action.")

        # Cast the args to the correct type
        if not isinstance(args, CleanupContextArgs):
            raise ValueError(f"Expected CleanupContextArgs, got {type(args)}")
        cleanup_args = args

        if not cleanup_args.file_paths:
            return Observation.create(
                message="No files specified for removal from context.",
                properties={"fail_reason": "empty_file_list"}
            )

        # Remove files from context
        removal_results = file_context.remove_files(cleanup_args.file_paths)
        
        # Separate successful and failed removals
        removed_files = [path for path, success in removal_results.items() if success]
        not_found_files = [path for path, success in removal_results.items() if not success]

        # Build response message
        message_parts = []
        
        if removed_files:
            if len(removed_files) == 1:
                message_parts.append(f"Successfully removed {removed_files[0]} from file context.")
            else:
                message_parts.append(f"Successfully removed {len(removed_files)} files from file context:")
                for file_path in removed_files:
                    message_parts.append(f"  - {file_path}")

        if not_found_files:
            if len(not_found_files) == 1:
                message_parts.append(f"File {not_found_files[0]} was not found in the current context.")
            else:
                message_parts.append(f"{len(not_found_files)} files were not found in the current context:")
                for file_path in not_found_files:
                    message_parts.append(f"  - {file_path}")

        message = "\n".join(message_parts)

        # Create summary
        if removed_files and not_found_files:
            summary = f"Removed {len(removed_files)} files, {len(not_found_files)} not found"
        elif removed_files:
            summary = f"Removed {len(removed_files)} files from context"
        else:
            summary = f"No files removed - {len(not_found_files)} not found in context"

        # Set properties for tracking
        properties = {
            "removed_count": len(removed_files),
            "not_found_count": len(not_found_files),
            "removed_files": removed_files,
            "not_found_files": not_found_files
        }

        # Determine if this should be considered a failure
        if not removed_files:
            properties["fail_reason"] = "no_files_removed"

        return Observation.create(
            message=message,
            summary=summary,
            properties=properties
        )

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> list[str]:
        return [
            "File Selection Relevance: Assess whether the files being removed are actually irrelevant to the current task.",
            "Context Management: Evaluate if removing these files helps focus the context on more relevant information.",
            "Task Understanding: Determine if the agent correctly identifies which files are no longer needed.",
            "Avoiding Over-cleanup: Check that the agent doesn't remove files that might still be useful for the task."
        ]