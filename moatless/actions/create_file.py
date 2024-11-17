import logging
from pathlib import Path
from typing import List

from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.actions.run_tests import RunTests, RunTestsArgs
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment

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

    class Config:
        title = "CreateFile"


class CreateFile(Action, CodeModificationMixin):
    """
    Action to create a new file with specified content.
    """

    args_schema = CreateFileArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        # Initialize mixin attributes directly
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_code_index", code_index)
        object.__setattr__(self, "_repository", repository)

    def execute(self, args: CreateFileArgs, file_context: FileContext) -> Observation:
        if args.path.startswith("/repo"):
            args.path = args.path[5:]
        if args.path.startswith("/"):
            args.path = args.path[1:]

        path = Path(args.path)

        if file_context.file_exists(str(path)):
            return Observation(
                message=f"File already exists at: {path}. Cannot overwrite files using create command.",
                properties={"fail_reason": "file_exists"},
            )

        context_file = file_context.add_file(str(path))
        context_file.apply_changes(args.file_text)

        diff = do_diff(str(path), "", args.file_text)

        observation = Observation(
            message=f"File created successfully at: {path}",
            properties={"diff": diff, "success": True},
        )

        if not self._runtime:
            return observation

        run_tests = RunTests(
            repository=self._repository,
            runtime=self._runtime,
            code_index=self._code_index,
        )
        test_observation = run_tests.execute(
            RunTestsArgs(
                scratch_pad=args.scratch_pad,
                test_files=[args.path],
            ),
            file_context,
        )

        observation.properties.update(test_observation.properties)
        observation.message += "\n\n" + test_observation.message

        return observation

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Create a new Python file for handling user authentication",
                action=CreateFileArgs(
                    scratch_pad="Creating a new authentication module with basic user authentication functionality",
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
            ),
            FewShotExample.create(
                user_input="Create a new configuration file",
                action=CreateFileArgs(
                    scratch_pad="Creating a configuration file with basic settings",
                    path="config/settings.py",
                    file_text="""from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = True

DATABASE = {
    'host': 'localhost',
    'port': 5432,
    'name': 'myapp_db',
    'user': 'admin'
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'level': 'INFO'
}""",
                ),
            ),
        ]
