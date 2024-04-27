import difflib
import logging
import os
from typing import List, Optional, Set

from pydantic import BaseModel

from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.module import Module
from moatless.codeblocks.parser.python import PythonParser

_parser = PythonParser()

logger = logging.getLogger(__name__)


class ContextFile(BaseModel):
    file_path: str
    module: Module
    span_ids: Set[str] = None

    def to_prompt(self, show_span_ids=False, show_line_numbers=False):
        code = self.module.to_prompt(
            show_span_id=show_span_ids,
            show_line_numbers=show_line_numbers,
            span_ids=self.span_ids,
            show_outcommented_code=False,
            exclude_block_types=[CodeBlockType.COMMENT],
        )

        return f"{self.file_path}\n```\n{code}\n```\n"


def do_diff(
    file_path: str, original_content: str, updated_content: str
) -> Optional[str]:
    return "".join(
        difflib.unified_diff(
            original_content.strip().splitlines(True),
            updated_content.strip().splitlines(True),
            fromfile=file_path,
            tofile=file_path,
            lineterm="\n",
        )
    )


class FileContext:

    def __init__(self, files: List[ContextFile], repo_path: str = None):
        self._parser = PythonParser()
        self._repo_path = repo_path
        self._files = {file.file_path: file for file in files}

    def is_in_context(self, file_path):
        return file_path in self._files

    def get_module(self, file_path):
        return self._files[file_path].module

    def update_module(self, file_path: str, module: Module):
        full_file_path = os.path.join(self._repo_path, file_path)
        logger.debug(f"Writing updated content to {full_file_path}")

        with open(full_file_path, "w") as f:
            f.write(module.to_string())

    def create_prompt(self, show_span_ids=False, show_line_numbers=False):
        file_context_content = ""
        for file in self._files.values():
            file_context_content += file.to_prompt(show_span_ids, show_line_numbers)
        return file_context_content
