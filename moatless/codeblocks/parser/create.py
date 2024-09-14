from moatless.codeblocks.parser.javascript import JavaScriptParser
from moatless.codeblocks.parser.parser import CodeParser
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.parser.java import JavaParser
from moatless.codeblocks.parser.typescript import TypeScriptParser


def is_supported(language: str) -> bool:
    return language and language in ["python", "java", "javascript", "jsx", "typescript", "tsx"]


def create_parser_by_ext(ext: str, **kwargs) -> CodeParser | None:
    if ext == ".py":
        return PythonParser(**kwargs)
    elif ext == ".java":
        return JavaParser(**kwargs)
    elif ext in [".js", ".jsx"]:
        return JavaScriptParser(**kwargs)
    elif ext in [".ts", ".tsx"]:
        return TypeScriptParser(**kwargs)

    raise NotImplementedError(f"Extension {ext} is not supported.")


def create_parser(language: str, **kwargs) -> CodeParser | None:
    if language == "python":
        return PythonParser(**kwargs)
    elif language == "java":
        return JavaParser(**kwargs)
    elif language in ["javascript", "jsx"]:
        return JavaScriptParser(**kwargs)
    elif language in ["typescript", "tsx"]:
        return TypeScriptParser(**kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
