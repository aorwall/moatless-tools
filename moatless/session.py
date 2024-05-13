from dataclasses import dataclass
from typing import List

from moatless.file_context import FileContext


# TOOD: How to handle the shared session object?


@dataclass
class _Session:

    _file_context: FileContext = None
    _session_id: str = None
    _tags: List[str] = None

    @property
    def file_context(self) -> FileContext:
        return self._file_context

    @file_context.setter
    def file_context(self, file_context: FileContext) -> None:
        self._file_context = file_context

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str) -> None:
        self._session_id = session_id

    @property
    def tags(self) -> List[str]:
        return list(self._tags) if self._tags else []

    @tags.setter
    def tags(self, tags: List[str]) -> None:
        self._tags = tags


Session = _Session()
