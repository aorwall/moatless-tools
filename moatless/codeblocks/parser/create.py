from moatless.codeblocks.parser.parser import CodeParser
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.parser.java import JavaParser


def is_supported(language: str) -> bool:
    return language and language in ["python", "java"]


def create_parser(language: str, **kwargs) -> CodeParser | None:
    if language == "python":
        return PythonParser(**kwargs)
    elif language == "java":
        return JavaParser(**kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
