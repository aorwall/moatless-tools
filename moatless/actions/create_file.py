import logging
from pathlib import Path

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.schema import ActionArguments, Observation
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)


class CreateFileArgs(ActionArguments):
    """
    Create a new file with specified content.

    Notes:
    * Cannot be used if the specified path already exists
    * Will create parent directories if they don't exist
    * File content should include proper indentation and formatting
    """

    path: str = Field(..., description="Path where the new file should be created")
    file_text: str = Field(..., description="Complete content to write to the new file")

    model_config = ConfigDict(title="CreateFile")

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<file_text>
{self.file_text}
</file_text>"""

    @classmethod
    def format_schema_for_llm(cls) -> str:
        return cls.format_xml_schema({"path": "file/path.py", "file_text": "\ncomplete file content\n"})

    @classmethod
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create a new Python file for handling user authentication",
                action=CreateFileArgs(
                    thoughts="Creating a new authentication module with basic user authentication functionality",
                    path="auth/user_auth.py",
                    file_text="""import logging
from typing import Optional

logger = logging.getLogger(__name__)

class UserAuth:
    def __init__(self):
        self._users = {}

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self._users:
            logger.warning(f"Authentication failed: User {username} not found")
            return False

        return self._users[username] == password

    def register(self, username: str, password: str) -> bool:
        if username in self._users:
            logger.error(f"Registration failed: User {username} already exists")
            return False

        self._users[username] = password
        logger.info(f"User {username} registered successfully")
        return True""",
                ),
            )
        ]


class CreateFile(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to create a new file with specified content.
    """

    args_schema = CreateFileArgs

    async def execute(
        self,
        args: CreateFileArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        if args.path.startswith("/repo"):
            args.path = args.path[5:]
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        if file_context.file_exists(str(path)):
            return Observation.create(
                message=f"File already exists at: {path}. Cannot overwrite files using create command.",
                properties={"fail_reason": "file_exists"},
            )

        context_file = file_context.add_file(str(path), show_all_spans=True)
        context_file.apply_changes(args.file_text)

        message = f"File created successfully at: {path}"

        test_summary = await self.run_tests(
            file_path=str(path),
            file_context=file_context,
        )

        if test_summary:
            message += f"\n\n{test_summary}"

        return Observation.create(message=message, summary=message)
