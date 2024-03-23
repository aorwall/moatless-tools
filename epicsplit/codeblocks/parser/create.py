from typing import Optional

from epicsplit.codeblocks.parser.java import JavaParser
from epicsplit.codeblocks.parser.javascript import JavaScriptParser
from epicsplit.codeblocks.parser.parser import CodeParser
from epicsplit.codeblocks.parser.python import PythonParser
from epicsplit.codeblocks.parser.typescript import TypeScriptParser


def is_supported(language: str) -> bool:
    return language and language in ["python", "java", "typescript", "javascript"]


def create_parser(language: str, **kwargs) -> Optional[CodeParser]:
    if language == "python":
        return PythonParser(**kwargs)

    if language == "java":
        return JavaParser(**kwargs)

    if language == "typescript" or language == "tsx":
        return TypeScriptParser("tsx", **kwargs)

    if language == "javascript":
        return JavaScriptParser(language, **kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
