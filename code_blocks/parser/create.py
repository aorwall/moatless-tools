from typing import Optional

from code_blocks.parser.java import JavaParser
from code_blocks.parser.parser import CodeParser
from code_blocks.parser.python import PythonParser


def create_parser(language: str) -> Optional[CodeParser]:
    if language == "java":
        return JavaParser()
    elif language == "python":
        return PythonParser()
    return None
