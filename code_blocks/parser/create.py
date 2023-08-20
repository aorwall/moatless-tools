from typing import Optional

from code_blocks.parser.java import JavaParser
from code_blocks.parser.parser import CodeParser
from code_blocks.parser.python import PythonParser


def create_parser(language: str, use_indentation_level: bool = False) -> Optional[CodeParser]:
    if language == "java":
        return JavaParser(use_indentation_level=use_indentation_level)
    elif language == "python":
        return PythonParser(gpt_mode=use_indentation_level)
    return None
