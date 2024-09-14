import logging

from tree_sitter import Language, Node
import tree_sitter_javascript as javascript
from tree_sitter_typescript import language_tsx

from moatless.codeblocks.codeblocks import CodeBlockType, CodeBlock, Relationship, ReferenceScope, RelationshipType
from moatless.codeblocks.parser.parser import CodeParser, NodeMatch

block_delimiters = [
    "{",
    "}",
    ";",
    "(",
    ")"
]

logger = logging.getLogger(__name__)

class JavaScriptParser(CodeParser):

    def __init__(self, language: str = "javascript", lang=Language(javascript.language()), **kwargs):
        if language == "javascript":
            lang = Language(javascript.language())
        elif language in ["tsx", "typescript"]:
            lang = Language(language_tsx())
        else:
            raise ValueError(f"Language {language} not supported")
        super().__init__(lang, **kwargs)
        self.queries = []

        if language in ["tsx", "typescript"]:
            self.queries.extend(self._build_queries("typescript.scm"))
        else:
            self.queries.extend(self._build_queries("javascript_only.scm"))

        self.queries.extend(self._build_queries("javascript.scm"))

        if self.apply_gpt_tweaks:
            self.queries.extend(self._build_queries("javascript_gpt.scm"))

    def process_match(self, node_match: NodeMatch, node: Node, content_bytes: bytes):
        if node.type in block_delimiters:  # TODO: Move to query
            node_match = NodeMatch(block_type=CodeBlockType.BLOCK_DELIMITER)

        return node_match

    def pre_process(self, codeblock: CodeBlock, node_match: NodeMatch):
        if codeblock.type == CodeBlockType.FUNCTION and codeblock.identifier == "constructor":
            codeblock.type = CodeBlockType.CONSTRUCTOR

        if codeblock.type == CodeBlockType.COMMENT and "..." in codeblock.content:
            codeblock.type = CodeBlockType.COMMENTED_OUT_CODE

        # Expect class on components with an identifier starting with upper case
        if codeblock.type == CodeBlockType.FUNCTION and codeblock.identifier and codeblock.identifier[0].isupper():
            codeblock.type = CodeBlockType.CLASS

        # Handle JSX closing elements as block delimiters
        if codeblock.content in ["/", ">"]:
            codeblock.type = CodeBlockType.BLOCK_DELIMITER
