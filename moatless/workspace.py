import json
import logging
import uuid
from typing import Optional, Any

from pydantic_core import to_jsonable_python

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository
from moatless.trajectory import Trajectory
from moatless.types import ActionRequest, FileWithSpans

_parser = PythonParser()

logger = logging.getLogger(__name__)


class Workspace:

    def __init__(
        self,
        file_repo: FileRepository,
        code_index: CodeIndex,
        workspace_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
    ):
        self._workspace_dir = workspace_dir
        self._workspace_id = workspace_id or str(uuid.uuid4())

        self._trajectory: Trajectory = None
        self._info = {}
        self._current_trajectory_step = None

        self.code_index = code_index
        self.file_repo = file_repo

        self._file_context = None

    @classmethod
    def from_dirs(
        cls,
        repo_dir: str,
        index_dir: str,
        workspace_dir: Optional[str] = None,
        **kwargs,
    ):
        file_repo = FileRepository(repo_dir)
        code_index = CodeIndex.from_persist_dir(index_dir, file_repo=file_repo)
        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            workspace_dir=workspace_dir,
            **kwargs,
        )
        return workspace

    def create_file_context(
        self, files_with_spans: Optional[list[FileWithSpans]] = None
    ):
        file_context = FileContext(self.file_repo)
        if files_with_spans:
            file_context.add_files_with_spans(files_with_spans)
        return file_context

    def get_file(self, file_path, refresh: bool = False):
        return self.file_repo.get_file(file_path, refresh=refresh)

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self.file_repo.save_file(file_path, updated_content)

    def save(self):
        self.file_repo.save()

    @property
    def trajectory(self):
        return self._trajectory

    def create_trajectory(self, name: str, input_data: Optional[dict[str, Any]] = None):
        if self._trajectory:
            return self._trajectory.create_child_trajectory(name, input_data=input_data)

        self._trajectory = Trajectory(name, input_data=input_data)
        return self._trajectory

    def new_trajectory_step(self, input: Optional[dict] = None):
        self._trajectory.new_step(input=input)

    def save_trajectory_thought(self, thought: str):
        self._trajectory.save_thought(thought)

    def save_trajectory_action(
        self, name: str, input: ActionRequest | dict, output: Optional[dict]
    ):
        self._trajectory.save_action(name, input, output)

    def save_trajectory_output(self, output: dict):
        self._trajectory.save_output(output)

    def save_trajectory_info(self, info: dict):
        self._trajectory.save_info(info)

    def save_trajectory(self, file_path: str):
        with open(f"{file_path}", "w") as f:
            f.write(
                json.dumps(
                    self._trajectory.dict(exclude_none=True),
                    indent=2,
                    default=to_jsonable_python,
                )
            )
