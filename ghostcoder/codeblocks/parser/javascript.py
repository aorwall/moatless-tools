import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

from tree_sitter import Node

from ghostcoder.codeblocks.codeblocks import CodeBlockType, CodeBlock
from ghostcoder.codeblocks.parser.parser import CodeParser, find_nested_type, find_type

class_node_types = [
    "class_declaration",
    "abstract_class_declaration",
    "enum_declaration",
    "interface_declaration"
]

function_node_types = [
    "method_definition",
    "function_declaration",
    "abstract_method_signature",
    "generator_function_declaration"
]

statement_node_types = [
    "if_statement",
    "for_statement",
    "try_statement",
    "return_statement"
]

block_delimiters = [
    "{",
    "}",
    ";",
    "(",
    ")"
]

logger = logging.getLogger(__name__)


@dataclass
class DefinitionTree:
    block_type: Optional[CodeBlockType] = None
    identifier: bool = False
    first_child: bool = False
    last_child: bool = False
    must_include_sub_types: bool = False
    first_child_index: int = None
    queries: List["MatchingQuery"] = field(default_factory=list)
    sub_tree: Dict[str, "DefinitionTree"] = field(default_factory=dict)


@dataclass
class MatchingQuery:
    query: str

    def __str__(self):
        return f"MatchingQuery(query={self.query})"


class_declaration = DefinitionTree(
    block_type=CodeBlockType.CLASS,
    sub_tree={
        "type_identifier": DefinitionTree(
            identifier=True,
        ),
        "identifier": DefinitionTree(
            identifier=True,
        ),
        "class_body": DefinitionTree(
            sub_tree={
                "{": DefinitionTree(
                    first_child=True
                )
            }
        )
    }
)

enum_declaration = DefinitionTree(
    block_type=CodeBlockType.CLASS,
    sub_tree={
        "type_identifier": DefinitionTree(
            identifier=True,
        ),
        "identifier": DefinitionTree(
            identifier=True,
        ),
        "enum_body": DefinitionTree(
            sub_tree={
                "{": DefinitionTree(
                    first_child=True
                )
            }
        )
    }
)
interface_declaration = DefinitionTree(
    block_type=CodeBlockType.CLASS,
    sub_tree={
            "type_identifier": DefinitionTree(
               identifier=True,
           ),
            "object_type": DefinitionTree(
                first_child_index=1,
            )
        }
    )

arrow_function = DefinitionTree(
    first_child_index=-1,
    sub_tree={
        "formal_parameters": DefinitionTree(
            block_type=CodeBlockType.FUNCTION
        )
    }
)

lexical_declaration = DefinitionTree(
    sub_tree={
        "variable_declarator": DefinitionTree(
            sub_tree={
                "identifier": DefinitionTree(
                    identifier=True,
                ),
                "type_annotation": DefinitionTree(
                    block_type=CodeBlockType.CLASS
                ),
                "arrow_function": arrow_function
            }
        )
    }
)

method_definition = DefinitionTree(
    block_type=CodeBlockType.FUNCTION,
    sub_tree={
        "property_identifier": DefinitionTree(
            identifier=True,
        ),
        "statement_block": DefinitionTree(
            first_child=True
        )
    }
)

variable_declarator = DefinitionTree(
    sub_tree={
        "identifier": DefinitionTree(
            identifier=True,
        ),
        "call_expression": DefinitionTree(
            must_include_sub_types=True,
            sub_tree={
                "identifier": DefinitionTree(
                ),
                "type_arguments": DefinitionTree(
                    block_type=CodeBlockType.CLASS
                ),
                "arguments": DefinitionTree(
                    sub_tree={
                        "arrow_function": arrow_function
                    }
                )
            }
        ),
        "arrow_function": arrow_function,
        "type_annotation": DefinitionTree(
            block_type=CodeBlockType.CODE
        ),
        "=": DefinitionTree(
            first_child=True
        )
    }
)

public_field_definition = DefinitionTree(
    sub_tree={
        "property_identifier": DefinitionTree(
            identifier=True,
        ),
        "arrow_function": arrow_function
    }
)

call_expression = DefinitionTree(
        sub_tree={
            "arguments": DefinitionTree(
                sub_tree={
                    "arrow_function": arrow_function,
                    #"(": DefinitionTree( #  FIXME
                    #    first_child=True,
                    #    block_type=CodeBlockType.CODE
                    #)
                }
            ),
            "member_expression": DefinitionTree(
                sub_tree={
                    "identifier": DefinitionTree(
                        identifier=True,
                    ),
                },
            )
        }
    )

_type_tree = {
    "class_declaration": class_declaration,
    "enum_declaration": enum_declaration,
    "interface_declaration": interface_declaration,
    "method_definition": method_definition,
    "public_field_definition": public_field_definition,
    "call_expression": call_expression,
    "variable_declaration": DefinitionTree(
        block_type=CodeBlockType.CODE,
        sub_tree={
            "variable_declarator": variable_declarator
        }
    ),
    "expression_statement": DefinitionTree(
        sub_tree={
            "call_expression": call_expression
        },
        queries=[
            MatchingQuery(
                query="""(expression_statement
  (assignment_expression
    (identifier) @identifier
    (arrow_function
      (formal_parameters) @type_function
      (statement_block) @first_child)))""",
            )
        ]
    ),
    "lexical_declaration": DefinitionTree(
        sub_tree={
            "variable_declarator": variable_declarator
        },
        queries=[
            MatchingQuery(
                query="""(lexical_declaration
  (variable_declarator
    [(identifier) @identifier
     (array_pattern) @identifier]
    [("=") @first_child @type_code
    (arrow_function
      (formal_parameters) @type_function
      [(statement_block) @first_child
       (expression) @first_child]
    )
    (call_expression
      (arguments
        (arrow_function
          (formal_parameters) @type_function
          (statement_block) @first_child
        )
      )
    )]
  )
)"""
            )
        ]
    ),
    "export_statement": DefinitionTree(
        sub_tree={
            "class_declaration": class_declaration,
            "interface_declaration": interface_declaration,
            "lexical_declaration": lexical_declaration,
            "enum_declaration": enum_declaration,
        }
    ),
    "binary_expression": DefinitionTree(
        block_type=CodeBlockType.CODE
    ),
    "new_expression": DefinitionTree(
        block_type=CodeBlockType.CODE
    ),
    "program": DefinitionTree(
        queries=[
            MatchingQuery(
                query="""(program
  (
  (expression_statement
    (call_expression 
      (identifier) @identifier @type_function
      (arguments)
    )
  )
  (statement_block) @first_child @last_child
  )
)""")],

        must_include_sub_types=True,
        sub_tree={
            "expression_statement": DefinitionTree(  # GPT corner case: constructor without a class
                sub_tree={
                    "call_expression": DefinitionTree(
                        sub_tree={
                            "identifier": DefinitionTree(
                                block_type=CodeBlockType.FUNCTION,
                                identifier=True
                            ),
                            "arguments": DefinitionTree(
                                sub_tree={
                                    "arrow_function": arrow_function
                                }
                            ),
                        },
                    )
                }
            ),
            "statement_block": DefinitionTree(
                first_child=True,
                last_child=True
            )
        }
    ),
}

class JavaScriptParser(CodeParser):

    def __init__(self, language: str = "javascript"):
        super().__init__(language)

    def find_in_tree(self, node: Node) -> Tuple[Optional[CodeBlockType], Optional[Node], Optional[Node], Optional[Node]]:
        if node.type in _type_tree:
            def_tree = _type_tree[node.type]
            for match_query in def_tree.queries:
                match_block_type, match_first_child, match_identifier_node, match_last_child = self.find_match(node, match_query)
                if match_block_type and match_identifier_node and match_first_child:
                    # logger.debug(f"Found match for {match_query} on node type {node.type}")
                    return match_block_type, match_first_child, match_identifier_node, match_last_child

            return self._find_in_tree(node, _type_tree)

        return None, None, None, None


    def _find_in_tree(self, node: Node, type_tree: dict) -> Tuple[Optional[CodeBlockType], Optional[Node], Optional[Node], Optional[Node]]:
        block_type = None
        first_child = None
        last_child = None
        identifier_node = None

        if node.type in type_tree:
            def_tree = type_tree[node.type]
            if def_tree.block_type:
                block_type = def_tree.block_type
            if def_tree.first_child:
                first_child = node
            if def_tree.last_child:
                last_child = node
            if def_tree.first_child_index is not None:
                first_child = node.children[def_tree.first_child_index]
            if def_tree.identifier:
                identifier_node = node

            for child in node.children:
                if def_tree.must_include_sub_types and child.type not in def_tree.sub_tree:
                    return None, None, None, None

                child_type, child_node, child_identifier_node, child_last_node = self._find_in_tree(child, def_tree.sub_tree)
                if not block_type:
                    block_type = child_type
                if child_node:
                    first_child = child_node
                if child_last_node:
                    last_child = child_last_node
                if not identifier_node:
                    identifier_node = child_identifier_node

        return block_type, first_child, identifier_node, last_child

    def find_arrow_func(self, node: Node):
        arrow_func = find_nested_type(node, "arrow_function", 2)
        if arrow_func:
            arrow = find_type(arrow_func, ["=>"])
            if arrow:
                block_delimiter = find_type(arrow.next_sibling, ["{"])
                if block_delimiter:
                    return block_delimiter
                else:
                    return arrow.next_sibling
        return None

    def find_match(self, node: Node, match_query: MatchingQuery) -> Tuple[CodeBlockType, Node, Node]:
        query = self.tree_language.query(match_query.query + " @root")
        captures = query.captures(node)

        identifier_node = None
        first_child = None
        block_type = None
        last_child = None

        for node_match, tag in captures:
            if tag == "root" and node != node_match:
                return None, None, None, None
            if tag == "identifier":
                identifier_node = node_match
            elif tag == "first_child":
                first_child = node_match
            elif tag == "last_child":
                last_child = node_match
            elif tag == "type_code":
                block_type = CodeBlockType.CODE
            elif tag == "type_class":
                block_type = CodeBlockType.CLASS
            elif tag == "type_function":
                block_type = CodeBlockType.FUNCTION

        return block_type, first_child, identifier_node, last_child

    def get_block_definition_2(self, node: Node, content_bytes: bytes, start_byte: int = 0) -> Tuple[Optional[CodeBlock], Optional[Node], Optional[Node]]:
        block_type, first_child, identifier_node, last_child = self.find_in_tree(node)
        if not block_type:
            return None, None, None

        if not last_child:
            if node.next_sibling and node.next_sibling.type == ";":
                last_child = node.next_sibling

            elif node.children:
                last_child = node.children[-1]

            if not first_child and last_child.type == ";":
                first_child = last_child

            if first_child and first_child.end_byte > last_child.end_byte:
                last_child = first_child

        pre_code = content_bytes[start_byte:node.start_byte].decode(self.encoding)
        end_line = node.end_point[0]

        if first_child:
            end_byte = self.get_previous(first_child, node)
        else:
            end_byte = node.end_byte

        code = content_bytes[node.start_byte:end_byte].decode(self.encoding)

        if identifier_node:
            identifier = content_bytes[identifier_node.start_byte:identifier_node.end_byte].decode(self.encoding)
        else:
            identifier = code.split("\n")[0].strip()

        if block_type == CodeBlockType.FUNCTION and identifier == "constructor":
            block_type = CodeBlockType.CONSTRUCTOR

        # A bit of a hack to support Jest tests
        if block_type == CodeBlockType.FUNCTION and code.startswith("describe("):
            block_type = CodeBlockType.TEST_SUITE

        if block_type == CodeBlockType.FUNCTION and code.startswith("it("):
            block_type = CodeBlockType.TEST_CASE

        # Workaround to set block type to class for React components with an identifier starting with upper case
        if block_type == CodeBlockType.FUNCTION and identifier[0].isupper():
            block_type = CodeBlockType.CLASS

        code_block = CodeBlock(
                type=block_type,
                identifier=identifier,
                tree_sitter_type=node.type,
                start_line=node.start_point[0],
                end_line=end_line,
                pre_code=pre_code,
                content=code,
                language=self.language,
                children=[]
            )

        return code_block, first_child, last_child

    def get_block_definition(self, node: Node) -> Tuple[CodeBlockType, Optional[Node], Optional[Node]]:
        if node.next_sibling and node.next_sibling.type == ";":
            last_child = node.next_sibling
        elif node.children:
            last_child = node.children[-1]

        if node.type == "program":
            return CodeBlockType.MODULE, node.children[0], last_child

        if node.type in function_node_types:
            return CodeBlockType.FUNCTION, find_nested_type(node, "{"), last_child

        if node.type in class_node_types:
            return CodeBlockType.CLASS, find_nested_type(node, "{"), last_child

        if node.type in block_delimiters:
            return CodeBlockType.BLOCK_DELIMITER, None, None

        if node.type == "import_statement":
            return CodeBlockType.IMPORT, find_type(node, ["import_clause"]), last_child

        if node.type == "import_clause":
            return CodeBlockType.CODE, find_nested_type(node, "import_specifier"), last_child

        if "comment" in node.type:
            if "..." in node.text.decode("utf8"):
                return CodeBlockType.COMMENTED_OUT_CODE, None, None
            else:
                return CodeBlockType.COMMENT, None, None

        if node.type in statement_node_types:
            block_type = CodeBlockType.STATEMENT
        else:
            block_type = CodeBlockType.CODE

        if node.type in ["binary_expression"]:
            return block_type, node.children[0], last_child

        if node.type in ["expression_statement"]:
            node = node.children[0]

        found_block_type, found_first_child, _, _ = self.find_in_tree(node)
        if found_block_type:
            return found_block_type, found_first_child, last_child

        if node.type == "lexical_declaration":
            node = find_type(node, ["variable_declarator"])

        if node.type in ["variable_declarator", "field_definition"]:
            arrow_function = find_type(node, ["arrow_function"])
            if arrow_function:
                formal_parameters = find_type(arrow_function, ["formal_parameters"])
                arrow = find_type(arrow_function, ["=>"])
                if arrow:
                    block_delimiter = find_nested_type(arrow.next_sibling, "{")
                    if block_delimiter:
                        first_child = block_delimiter
                    else:
                        first_child = arrow.next_sibling

                type_annotation = find_nested_type(node, "type_annotation")
                if type_annotation and type_annotation.start_byte < first_child.start_byte:
                    return CodeBlockType.CLASS, first_child, last_child
                elif formal_parameters:
                    return CodeBlockType.FUNCTION, first_child, last_child
                else:
                    return CodeBlockType.STATEMENT, first_child, last_child

        if node.type in ["variable_declarator", "variable_declaration", "call_expression",
                         "new_expression", "type_alias_declaration", "field_definition"]:
            delimiter = find_type(node, ["="])
            if delimiter:
                return block_type, delimiter, last_child

            arrow_func = self.find_arrow_func(node)
            if arrow_func:
                return block_type, arrow_func, last_child

            return block_type, node.children[0], last_child

        if node.type in ["object"]:
            return block_type, find_type(node, ["{"]).next_sibling, last_child

        if node.type in ["jsx_element"]:
            return CodeBlockType.STATEMENT, node.children[0], last_child

        if node.type in ["jsx_opening_element"]:
            jsx_attr = find_type(node, ["jsx_attribute"])
            if jsx_attr:
                return block_type, jsx_attr, last_child
            else:
                return block_type, None, None

        block = find_type(node,
                          ["class_body", "enum_body", "statement_block", "object_type", "object_pattern", "switch_body",
                           "jsx_expression"])
        if block:
            delimiter = find_type(block, ["{"])
            if delimiter:
                return block_type, delimiter, last_child
            return block_type, block.children[0], last_child

        call_func = find_type(node, ["call_expression"])
        if call_func:
            arrow_func = self.find_arrow_func(call_func);
            if arrow_func:
                return block_type, arrow_func, last_child

        if node.type == "parenthesized_expression":
            parenthesized_expression = node
        else:
            parenthesized_expression = find_type(node, ["parenthesized_expression"])
        if parenthesized_expression:
            delimiter = find_type(parenthesized_expression, ["("])
            if delimiter:
                return block_type, delimiter, last_child
            return block_type, parenthesized_expression.children[0], last_child

        # start children after :
        if node.type in ["switch_case", "switch_default", "pair"]:
            delimiter = find_type(node, [":"])
            if delimiter:
                return block_type, delimiter, last_child

        if node.type in ["statement_block"]:
            return block_type, node.children[0], last_child

        if node.type in ["return_statement", "call_expression", "assignment_expression"]:
            return block_type, node.children[1], last_child

        return CodeBlockType.CODE, None, None
