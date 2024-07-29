import logging
from typing import Optional

from moatless.codeblocks.parser.python import PythonParser
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository import FileRepository
from moatless.types import FileWithSpans

_parser = PythonParser()

logger = logging.getLogger(__name__)

from .utils_search.deepcopy import safe_deepcopy


class Workspace:

    def __init__(
        self,
        file_repo: FileRepository,
        code_index: Optional[CodeIndex] = None,
        repo_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
    ):
        self.code_index = code_index
        self.file_repo = file_repo
        self._file_context = self.create_file_context()
        self.repo_dir = repo_dir
        self.index_dir = index_dir

    @classmethod
    def from_dirs(
        cls,
        repo_dir: str,
        index_dir: Optional[str] = None,
    ):
        file_repo = FileRepository(repo_dir)
        if index_dir:
            code_index = CodeIndex.from_persist_dir(index_dir, file_repo=file_repo)
        else:
            code_index = None
        
        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            repo_dir=repo_dir,
            index_dir=index_dir,
        )
        return workspace

    def create_file_context(
        self, files_with_spans: Optional[list[FileWithSpans]] = None
    ):
        file_context = FileContext(self.file_repo)
        if files_with_spans:
            file_context.add_files_with_spans(files_with_spans)
        return file_context

    @property
    def file_context(self):
        return self._file_context

    def get_file(self, file_path, refresh: bool = False, from_origin: bool = False):
        return self.file_repo.get_file(file_path, refresh=refresh, from_origin=from_origin)

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self.file_repo.save_file(file_path, updated_content)

    def save(self):
        self.file_repo.save()
        
    def serialize(self):
        """
        Serialize the Workspace into a dictionary.
        """
        serialized = {
            "file_repo": self.file_repo.serialize(),
            "repo_dir": self.repo_dir,
            "index_dir": self.index_dir,
            "file_context": self._file_context.serialize() if self._file_context else None,
        }
        
        if self.code_index:
            serialized["code_index"] = self.code_index.serialize()
        else:
            serialized["code_index"] = None

        return serialized

    @classmethod
    def deserialize(cls, serialized_data):
        """
        Deserialize the Workspace from a dictionary and create a new instance.
        """
        file_repo = FileRepository.deserialize(serialized_data["file_repo"])
        
        if serialized_data["code_index"]:
            code_index = CodeIndex.deserialize(serialized_data["code_index"])
        else:
            code_index = None

        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            repo_dir=serialized_data["repo_dir"],
            index_dir=serialized_data["index_dir"],
        )

        if serialized_data["file_context"]:
            workspace._file_context = FileContext.deserialize(serialized_data["file_context"])
        else:
            workspace._file_context = None

        return workspace
        
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        
        # # print current workspace for debugging
        # print("Current workspace:")
        # print(f"{self.file_context.files}")
        # print(f"----------------")

        # Create a new Workspace instance
        new_workspace = safe_deepcopy(Workspace.__new__(Workspace))
        memo[id(self)] = new_workspace

        # Deepcopy FileRepository
        new_file_repo = FileRepository(self.file_repo._repo_path)
        new_file_repo._files = safe_deepcopy(self.file_repo._files, memo)

        # safe_deepcopy CodeIndex if it exists
        if self.code_index:
            new_code_index = safe_deepcopy(self.code_index, memo)
        else:
            new_code_index = None

        # Set attributes
        new_workspace.file_repo = new_file_repo
        new_workspace.code_index = new_code_index

        # safe_deepcopy FileContext
        new_workspace._file_context = safe_deepcopy(self._file_context, memo)

        # Ensure the new FileContext uses the new FileRepository
        new_workspace._file_context._repo = new_file_repo

        return new_workspace
    
    def serialize(self):
        """
        Serialize the Workspace into a dictionary.
        """
        serialized = {
            "file_repo": self.file_repo.serialize(),
            "repo_dir": self.repo_dir,
            "index_dir": self.index_dir,
            "file_context": self._file_context.serialize() if self._file_context else None,
        }
        
        if self.code_index:
            serialized["code_index"] = self.code_index.serialize()
        else:
            serialized["code_index"] = None

        return serialized

    @classmethod
    def deserialize(cls, serialized_data):
        """
        Deserialize the Workspace from a dictionary and create a new instance.
        """
        file_repo = FileRepository.deserialize(serialized_data["file_repo"])
        
        if serialized_data["code_index"]:
            code_index = CodeIndex.deserialize(serialized_data["code_index"])
        else:
            code_index = None

        workspace = cls(
            file_repo=file_repo,
            code_index=code_index,
            repo_dir=serialized_data["repo_dir"],
            index_dir=serialized_data["index_dir"],
        )

        if serialized_data["file_context"]:
            workspace._file_context = FileContext.deserialize(serialized_data["file_context"])
        else:
            workspace._file_context = None

        return workspace