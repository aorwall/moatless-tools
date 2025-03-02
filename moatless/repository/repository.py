import importlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Repository(BaseModel, ABC):
    @abstractmethod
    def get_file_content(self, file_path: str) -> Optional[str]:
        pass

    def file_exists(self, file_path: str) -> bool:
        return True

    def save_file(self, file_path: str, updated_content: str):
        pass

    def is_directory(self, file_path: str) -> bool:
        return False

    def model_dump(self, **kwargs) -> dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["repository_class"] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "Repository":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository_class_path = obj.pop("repository_class", None)

            if repository_class_path:
                module_name, class_name = repository_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                repository_class = getattr(module, class_name)
                instance = repository_class(**obj)
            else:
                return None

            return instance

        return super().model_validate(obj)

    @abstractmethod
    def list_directory(self, directory_path: str = "") -> dict[str, list[str]]:
        """
        Lists files and directories in the specified directory.
        Returns a dictionary with 'files' and 'directories' lists.
        """
        pass


class InMemRepository(Repository):
    files: dict[str, str] = Field(default_factory=dict)

    def __init__(self, files: dict[str, str] = None, **kwargs):
        files = files or {}
        super().__init__(files=files, **kwargs)

    def get_file_content(self, file_path: str) -> Optional[str]:
        return self.files.get(file_path)

    def file_exists(self, file_path: str) -> bool:
        return file_path in self.files

    def save_file(self, file_path: str, updated_content: str):
        self.files[file_path] = updated_content

    def is_directory(self, file_path: str) -> bool:
        return False

    def list_directory(self, directory_path: str = "") -> dict[str, list[str]]:
        return {"files": [], "directories": []}

    def model_dump(self) -> dict:
        return {"files": self.files}

    @classmethod
    def model_validate(cls, obj: dict):
        return cls(files=obj.get("files", {}))
