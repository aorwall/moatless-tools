from typing import Optional

from moatless.codeblocks.parser.parser import CodeParser
from moatless.codeblocks.parser.python import PythonParser


def is_supported(language: str) -> bool:
    return language and language in ["python", "java", "typescript", "javascript"]


def create_parser(language: str, **kwargs) -> Optional[CodeParser]:
    if language == "python":
        return PythonParser(**kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
