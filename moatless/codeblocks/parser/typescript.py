from tree_sitter import Language
from tree_sitter_typescript import language_tsx

from moatless.codeblocks.parser.javascript import JavaScriptParser


class TypeScriptParser(JavaScriptParser):

    def __init__(self, language: str = "tsx", **kwargs):
        super().__init__(language, lang=Language(language_tsx()), **kwargs)
