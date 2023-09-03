from typing import Optional

from codeblocks_gpt.parser.java import JavaParser
from codeblocks_gpt.parser.parser import CodeParser
from codeblocks_gpt.parser.python import PythonParser
from codeblocks_gpt.parser.typescript import TypeScriptParser


def create_parser(language: str) -> Optional[CodeParser]:
    if language == "java":
        return JavaParser()
    elif language == "python":
        return PythonParser()
    elif language == "typescript" or language == "tsx":
        return TypeScriptParser(language)
    else:
        return CodeParser(language)
