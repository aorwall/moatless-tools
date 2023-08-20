from typing import Optional

from codeblocks.parser.java import JavaParser
from codeblocks.parser.parser import CodeParser
from codeblocks.parser.python import PythonParser


def create_parser(language: str) -> Optional[CodeParser]:
    if language == "java":
        return JavaParser()
    elif language == "python":
        return PythonParser()
    return None
