import difflib
import logging
import os
from typing import Optional, Set, Dict

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

    def to_prompt(
        self, show_span_ids=False, show_line_numbers=False, exclude_comments=False
    ):
        exclude_types = [CodeBlockType.COMMENT] if exclude_comments else []
        code = self.module.to_prompt(
            show_span_id=show_span_ids,
            show_line_numbers=show_line_numbers,
            span_ids=self.span_ids,
            show_outcommented_code=False,
            exclude_block_types=exclude_types,
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

    def __init__(self, repo_path: str):
        self._parser = PythonParser()
        self._repo_path = repo_path
        self._files: Dict[str, ContextFile] = {}

    def add_to_context(self, file_path: str, span_ids: Set[str]):
        file = self._files.get(file_path)
        if not file:
            with open(os.path.join(self._repo_path, file_path), "r") as f:
                content = f.read()
                module = _parser.parse(content)
                file = ContextFile(file_path=file_path, module=module)
                self._files[file_path] = file

        file.span_ids = span_ids

    def get_file(self, file_path):
        file = self._files.get(file_path)
        if not file:
            if not os.path.exists(os.path.join(self._repo_path, file_path)):
                return None
            with open(os.path.join(self._repo_path, file_path), "r") as f:
                content = f.read()
                module = _parser.parse(content)
                file = ContextFile(file_path=file_path, module=module)
                self._files[file_path] = file

        return file

    def save_file(self, file_path: str, updated_content: str = None):
        file = self._files.get(file_path)

        full_file_path = os.path.join(self._repo_path, file.file_path)
        logger.debug(f"Writing updated content to {full_file_path}")

        with open(full_file_path, "w") as f:
            updated_content = updated_content or file.module.to_string()
            f.write(updated_content)

    def create_prompt(
        self, show_span_ids=False, show_line_numbers=False, exclude_comments=False
    ):
        file_context_content = ""
        for file in self._files.values():
            file_context_content += file.to_prompt(
                show_span_ids, show_line_numbers, exclude_comments
            )
        return file_context_content

    def dict(self):
        file_dict = []
        for file_path, file in self._files.items():
            file_dict.append({"file_path": file_path, "span_ids": list(file.span_ids)})
        return file_dict
