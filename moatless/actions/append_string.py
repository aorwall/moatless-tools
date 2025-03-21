import re

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.schema import ActionArguments, Observation
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.repository.file import do_diff


class AppendStringArgs(ActionArguments):
    """
    Append text content to the end of a file.
    """

    model_config = ConfigDict(title="AppendString")

    path: str = Field(..., description="Path to the file to append to")
    new_str: str = Field(..., description="Text content to append at the end of the file")

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<new_str>
{self.new_str}
</new_str>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return cls.format_xml_schema({"path": "file/path.py", "new_str": "\ncontent to append at end of file\n"})

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Add a new helper function at the end of the utilities file",
                action=AppendStringArgs(
                    thoughts="Adding a new utility function for date formatting",
                    path="utils/formatters.py",
                    new_str="""

def format_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()""",
                ),
            ),
        ]


class AppendString(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to append text content strictly to the end of a file.
    This action only adds content at the file's end and cannot modify existing content.
    """

    args_schema = AppendStringArgs

    async def execute(
        self,
        args: AppendStringArgs,
        file_context: FileContext | None = None,
    ) -> Observation:
        if not file_context:
            raise RuntimeError("File context is not set")

        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        if not context_file:
            return Observation.create(
                message=f"Could not get context for file: {path}",
                properties={"fail_reason": "context_error"},
            )

        file_text = context_file.content.expandtabs()
        new_str = args.new_str.expandtabs()

        # Check if this looks like top-of-file content (imports, etc)
        looks_like_import = bool(re.match(r"^(import|from)\s+\w+", new_str.lstrip()))

        if looks_like_import:
            return Observation.create(
                message=(
                    "It looks like you're trying to add imports or other top-of-file content. "
                    "Please use StringReplace action to add content at the beginning of files."
                ),
                properties={"fail_reason": "wrong_action_for_imports"},
            )

        # Normal append logic
        if file_text:
            file_text = file_text.rstrip("\n")
            new_str = "\n\n" + new_str.lstrip("\n")
        else:
            new_str = new_str.lstrip("\n")

        new_file_text = file_text + new_str

        diff = do_diff(str(path), file_text, new_file_text)
        context_file.apply_changes(new_file_text)

        message = (
            f"The file {path} has been edited. "
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
        )

        return Observation.create(
            message=message,
            summary=message,
            properties={"diff": diff, "success": True},
        )
