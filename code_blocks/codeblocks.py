from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CodeBlockType(str, Enum):
    DECLARATION = "declaration"
    IDENTIFIER = "identifier"
    PARAMETER = "parameter"

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"

    STATEMENT = "statement"
    CODE = "code"
    BLOCK_DELIMITER = "block_delimiter"

    COMMENT = "comment"
    COMMENTED_OUT_CODE = "commented_out_code"

    SPACE = "space"
    ERROR = "error"


@dataclass
class CodeBlock:
    content: str
    type: CodeBlockType
    content_lines: List[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    tree_sitter_type: Optional[str] = None
    pre_code: str = ""
    pre_lines: int = 0
    indentation: str = ""

    children: List["CodeBlock"] = field(default_factory=list)
    parent: Optional["CodeBlock"] = field(default=None)

    def __post_init__(self):
        for child in self.children:
            child.parent = self

        pre_code_lines = self.pre_code.split("\n")
        self.pre_lines = len(pre_code_lines) - 1
        if self.pre_lines > 0:
            self.indentation = pre_code_lines[-1]

        self.content_lines = self.content.split("\n")
        if self.indentation and self.pre_lines:
            self.content_lines[1:] = [line[len(self.indentation):] for line in self.content_lines[1:]]

    def insert_child(self, index: int, child: "CodeBlock"):
        if index == 0 and self.children[0].pre_lines == 0:
            self.children[0].pre_lines = 1

        self.children.insert(index, child)
        child.parent = self

    def insert_children(self, index: int, children: List["CodeBlock"]):
        for child in children:
            self.insert_child(index, child)
            index += 1

    def __str__(self):
        return str(self.to_dict())

    def to_string(self):
        child_code = "".join([child.to_string() for child in self.children])

        if self.pre_lines:
            linebreaks = "\n" * self.pre_lines
            content = linebreaks + "\n".join(self.indentation + line for line in self.content_lines)
        else:
            content = self.pre_code + self.content

        return f"{content}{child_code}"

    def to_tree(self, indent: int = 0):
        child_code = "".join([child.to_tree(indent+1) for child in self.children])
        indent_str = " " * indent
        return f"\n{indent_str} {indent}. {self.type.value} : {self.content}{child_code}"

    def __eq__(self, other):
        if not isinstance(other, CodeBlock):
            return False
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        return {
            "code": self.content,
            "type": self.type,
            "tree_sitter_type": self.tree_sitter_type,
            "pre_code": self.pre_code,
            #"is_nested": self.is_nested,
            "children": [child.to_dict() for child in self.children]
        }

    def root(self):
        if self.parent:
            return self.parent.root()
        return self

    def is_complete(self):
        if self.type == CodeBlockType.COMMENTED_OUT_CODE:
            return False
        for child in self.children:
            if not child.is_complete():
                return False
        return True

    def find_errors(self) -> List["CodeBlock"]:
        errors = []
        if self.children:
            for child in self.children:
                errors.extend(child.find_errors())

        if self.type == CodeBlockType.ERROR:
            errors.append(self)

        return errors

    def add_indentation(self, indentation: str):
        if self.pre_lines:
            self.indentation += indentation
        for child in self.children:
            child.add_indentation(indentation)

    def find_equal_parent(self, check_block: "CodeBlock") -> Optional["CodeBlock"]:
        if not self.parent:
            return None

        if self.parent == check_block:
            return self

        return self.parent.find_equal_parent(check_block)

    def has_matching_child(self, other_children: List["CodeBlock"], start_original: int = 0) -> bool:
        original_contents = [child.content for child in self.children[start_original:]]
        updated_contents = [child.content for child in other_children]
        return any(ub_content in original_contents for ub_content in updated_contents)

    def find_next_matching_block(self, children_start, other_child_block):
        i = children_start
        while i < len(self.children):
            check_original_block = self.children[i]
            if (check_original_block.content
                    and check_original_block.content == other_child_block.content
                    and check_original_block.type == other_child_block.type):
                return i
            i += 1
        return None

    def find_nested_matching_block(self, other: "CodeBlock", start_original: int = 0) -> Optional["CodeBlock"]:
        for child_block in self.children[start_original:]:
            if child_block.children:
                for other_child_block in other.children:
                    if child_block.content == other_child_block.content and child_block.type == other_child_block.type:
                        return child_block

                nested_match = child_block.find_nested_matching_block(other)
                if nested_match:
                    return nested_match
        return None

    def find_next_commented_out(self, start):
        i = start
        while i < len(self.children):
            if self.children[i].type == CodeBlockType.COMMENTED_OUT_CODE:
                return i
            i += 1
        return None
