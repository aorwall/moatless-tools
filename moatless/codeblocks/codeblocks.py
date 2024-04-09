import copy
import logging
import re
from collections import namedtuple
from enum import Enum
from typing import List, Optional, Callable, Tuple

from pydantic import BaseModel, validator, Field, root_validator

from moatless.codeblocks.parser.comment import get_comment_symbol
from moatless.codeblocks.utils import Colors
from moatless.types import BlockPath


class CodeBlockType(str, Enum):
    DECLARATION = "Declaration"
    IDENTIFIER = "Identifier"
    PARAMETER = "Parameter"

    MODULE = "Module"
    CLASS = "Class"
    FUNCTION = "Function"
    CONSTRUCTOR = "Constructor"

    TEST_SUITE = "TestSuite"
    TEST_CASE = "TestCase"

    IMPORT = "Import"
    EXPORT = "Export"
    STATEMENT = "Statement"
    BLOCK = "Block"
    ASSIGNMENT = "Assignment"
    CALL = "Call"
    CODE = "Code"
    BLOCK_DELIMITER = "BlockDelimiter"

    COMMENT = "Comment"
    COMMENTED_OUT_CODE = "CommentedOutCode"  # TODO: Replace to PlaceholderComment

    SPACE = "Space"
    ERROR = "Error"

    @classmethod
    def from_string(cls, tag: str) -> Optional['CodeBlockType']:
        if not tag.startswith("definition"):
            return None

        tag_to_block_type = {
            "definition.assignment": cls.ASSIGNMENT,
            "definition.block": cls.BLOCK,
            "definition.block_delimiter": cls.BLOCK_DELIMITER,
            "definition.call": cls.CALL,
            "definition.class": cls.CLASS,
            "definition.code": cls.CODE,
            "definition.comment": cls.COMMENT,
            "definition.constructor": cls.CONSTRUCTOR,
            "definition.error": cls.ERROR,
            "definition.export": cls.EXPORT,
            "definition.function": cls.FUNCTION,
            "definition.import": cls.IMPORT,
            "definition.module": cls.MODULE,
            "definition.statement": cls.STATEMENT,
            "definition.test_suite": cls.TEST_SUITE,
            "definition.test_case": cls.TEST_CASE,
        }
        return tag_to_block_type.get(tag)


NON_CODE_BLOCKS = [
    CodeBlockType.BLOCK_DELIMITER,
    CodeBlockType.COMMENT,
    CodeBlockType.COMMENTED_OUT_CODE,
    CodeBlockType.EXPORT,
    CodeBlockType.IMPORT,
    CodeBlockType.ERROR,
    CodeBlockType.SPACE
]

INDEXED_BLOCKS = [
    CodeBlockType.FUNCTION,
    CodeBlockType.CLASS,
    CodeBlockType.TEST_SUITE,
    CodeBlockType.TEST_CASE
]


Span = namedtuple('Span', ['start_line', 'end_line'])


class PathTree(BaseModel):
    show: bool = Field(default=False, description="Show the block and all sub blocks.")
    tree: dict[str, 'PathTree'] = Field(default_factory=dict)

    @staticmethod
    def from_block_paths(block_paths: List[BlockPath]) -> 'PathTree':
        tree = PathTree()
        for block_path in block_paths:
            tree.add_to_tree(block_path)

        return tree

    def child_tree(self, key: str) -> Optional['PathTree']:
        return self.tree.get(key, None)

    def merge(self, other: 'PathTree'):
        if other.show:
            self.show = True

        for key, value in other.tree.items():
            if key not in self.tree:
                self.tree[key] = PathTree()
            self.tree[key].merge(value)

    def extend_tree(self, paths: list[list[str]]):
        for path in paths:
            self.add_to_tree(path)

    def add_to_tree(self, path: list[str]):
        if path is None:
            return

        if len(path) == 0:
            self.show = True
            return

        if len(path) == 1:
            if path[0] not in self.tree:
                self.tree[path[0]] = PathTree(show=True)
            else:
                self.tree[path[0]].show = True

            return

        if path[0] not in self.tree:
            self.tree[path[0]] = PathTree(show=False)

        self.tree[path[0]].add_to_tree(path[1:])

class ReferenceScope(str, Enum):
    EXTERNAL = "external"
    DEPENDENCY = "dependency"  # External dependency
    FILE = "file"  # File in repository
    PROJECT = "project"
    CLASS = "class"
    LOCAL = "local"
    GLOBAL = "global"


class RelationshipType(str, Enum):
    UTILIZES = "utilizes"
    USES = "uses"
    DEFINED_BY = "defined_by"
    IS_A = "is_a"
    PROVIDES = "provides"
    IMPORTS = "imports"
    CALLS = "calls"
    DEPENDENCY = "dependency"
    TYPE = "type"


class DiffAction(Enum):
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"


class BlockDiff(BaseModel):
    block_path: List[str] = Field()
    action: DiffAction = Field()


class MergeAction(BaseModel):
    action: str
    original_block: Optional["CodeBlock"] = None
    updated_block: Optional["CodeBlock"] = None

    def __str__(self):
        original_id = self.original_block.identifier if self.original_block else 'None'
        updated_id = self.updated_block.identifier if self.updated_block else 'None'
        return f"{self.action}: {original_id or '[]'} -> {updated_id or '[]'}"


class Relationship(BaseModel):
    scope: ReferenceScope = Field(description="The scope of the reference.")
    identifier: Optional[str] = Field(default=None, description="ID")
    type: RelationshipType = Field(default=RelationshipType.USES, description="The type of the reference.")
    external_path: List[str] = Field(default=[], description="The path to the referenced parent code block.")
    resolved_path: List[str] = Field(default=[], description="The path to the file with the referenced code block.")
    path: List[str] = Field(default=[], description="The path to the referenced code block.")

    @root_validator(pre=True)
    def validate_path(cls, values):
        external_path = values.get("external_path")
        path = values.get("path")
        if not external_path and not path:
            raise ValueError("Cannot create Reference without external_path or path.")
        return values

    def __hash__(self):
        return hash((self.scope, tuple(self.path)))

    def __eq__(self, other):
        return (self.scope, self.path) == (other.scope, other.path)

    def full_path(self):
        return self.external_path + self.path

    def __str__(self):
        if self.identifier:
            start_node = self.identifier
        else:
            start_node = ""

        end_node = ""
        if self.external_path:
            end_node = "/".join(self.external_path)
        if self.path:
            if self.external_path:
                end_node += "/"
            end_node += ".".join(self.path)

        return f"({start_node})-[:{self.type.name} {{scope: {self.scope.value}}}]->({end_node})"


class Parameter(BaseModel):
    identifier: str = Field(description="The identifier of the parameter.")
    type: Optional[str] = Field(description="The type of the parameter.")


class Visibility(str, Enum):
    EVERYTHING = "everything"  # Show all blocks
    CODE = "code"  # Only show code but comment out function and class definitions
    NOTHING = "nothing"  # Comment out everything


class ShowBlock(BaseModel):
    visibility: Visibility = Field(default=Visibility.EVERYTHING, description="The visibility of the block.")
    tree: dict[str, "ShowBlock"] = Field(default={}, description="Child blocks")


class CodeBlock(BaseModel):
    content: str
    type: CodeBlockType
    file_path: str = None  # TODO: Move to Module sub class
    identifier: Optional[str] = None
    is_indexed: bool = False
    parameters: List[Parameter] = []  # TODO: Move to Function sub class
    references: List[Relationship] = []
    content_lines: List[str] = []
    start_line: int = 0
    end_line: int = 0
    properties: dict = {}
    pre_code: str = ""
    pre_lines: int = 0
    indentation: str = ""
    tokens: int = 0
    language: Optional[str] = None
    children: List["CodeBlock"] = []
    parent: Optional["CodeBlock"] = None

    @validator('type', pre=True, always=True)
    def validate_type(cls, v):
        if v is None:
            raise ValueError("Cannot create CodeBlock without type.")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        for child in self.children:
            child.parent = self

        if self.pre_code and not re.match(r"^[ \n\\]*$", self.pre_code):
            raise ValueError(f"Failed to parse code block with type {self.type} and content `{self.content}`. "
                             f"Expected pre_code to only contain spaces and line breaks. Got `{self.pre_code}`")

        if self.pre_code and not self.indentation and not self.pre_lines:
            pre_code_lines = self.pre_code.split("\n")
            self.pre_lines = len(pre_code_lines) - 1
            if self.pre_lines > 0:
                self.indentation = pre_code_lines[-1]
            else:
                self.indentation = self.pre_code

        self.content_lines = self.content.split("\n")
        #if self.indentation and self.pre_lines:
        #    self.content_lines[1:] = [line[len(self.indentation):] for line in self.content_lines[1:]]

    def insert_child(self, index: int, child: "CodeBlock"):
        if index == 0 and self.children[0].pre_lines == 0:
            self.children[0].pre_lines = 1

        self.children.insert(index, child)
        child.parent = self

    def insert_children(self, index: int, children: List["CodeBlock"]):
        for child in children:
            self.insert_child(index, child)
            index += 1

    def append_child(self, child: "CodeBlock"):
        self.children.append(child)
        child.parent = self

    def append_children(self, children: List["CodeBlock"]):
        for child in children:
            self.append_child(child)

    def replace_child(self, index: int, child: "CodeBlock"):
        # TODO: Do a proper update of everything when replacing child blocks
        child.pre_code = self.children[index].pre_code
        child.pre_lines = self.children[index].pre_lines
        self.sync_indentation(self.children[index], child)

        self.children[index] = child
        child.parent = self

    def sync_indentation(self, original_block: "CodeBlock", updated_block: "CodeBlock"):
        original_indentation_length = len(original_block.indentation) + len(self.indentation)
        updated_indentation_length = len(updated_block.indentation) + len(updated_block.parent.indentation)

        # To handle separate code blocks provdided out of context
        if original_indentation_length == updated_indentation_length and len(updated_block.indentation) == 0:
            updated_block.indentation = ' ' * original_indentation_length

        elif original_indentation_length > updated_indentation_length:
            additional_indentation = ' ' * (original_indentation_length - updated_indentation_length)
            updated_block.add_indentation(additional_indentation)

    def replace_children(self, index: int, children: List["CodeBlock"]):
        for child in children:
            self.replace_child(index, child)
            index += 1

    def remove_child(self, index: int):
        del self.children[index]

    def replace_by_path(self, path: List[str], new_block: "CodeBlock"):
        if not path:
            return

        for i, child in enumerate(self.children):
            if child.identifier == path[0]:
                if len(path) == 1:
                    self.replace_child(i, new_block)
                else:
                    child.replace_by_path(path[1:], new_block)

    def has_equal_definition(self, other: "CodeBlock") -> bool:
        # TODO: should be replaced an expression that checks the actual identifier and parameters
        return other.content in self.content and other.type == self.type

    def __str__(self):
        return self.to_string()

    def to_string(self):
        return self._to_string()

    def _insert_into_tree(self, node, path):
        if not path:
            return
        if path[0] not in node:
            node[path[0]] = {}
        self._insert_into_tree(node[path[0]], path[1:])

    def _replace_in_tree(self, node, path):
        if not path:
            return
        if len(path) == 1:
            node[path[0]] = {}
            return
        if path[0] not in node:
            node[path[0]] = {}
        self._replace_in_tree(node[path[0]], path[1:])

    def sum_tokens(self):
        tokens = self.tokens
        tokens += sum([child.sum_tokens() for child in self.children])
        return tokens

    def get_all_child_blocks(self) -> List["CodeBlock"]:
        blocks = []
        for child in self.children:
            blocks.append(child)
            blocks.extend(child.get_all_child_blocks())
        return blocks

    def get_children(self, exclude_blocks: List[CodeBlockType] = []) -> List["CodeBlock"]:
        return [child for child in self.children if child.type not in exclude_blocks]

    def _to_string(self) -> str:
        contents = ""

        if self.pre_lines:
            contents += "\n" * (self.pre_lines - 1)
            for i, line in enumerate(self.content_lines):
                if i == 0 and line:
                    contents += "\n" + self.indentation + line
                elif line:
                    contents += "\n" + line
                else:
                    contents += "\n"
        else:
            contents += self.pre_code + self.content

        for i, child in enumerate(self.children):
            contents += child._to_string()

        return contents

    def _build_path_tree(self, block_paths: List[str], include_references: bool = False):
        path_tree = PathTree()

        for block_path in block_paths:
            if block_path:
                path = block_path.split(".")
                if include_references:
                    block = self.find_by_path(path)
                    if block:
                        if self.type == CodeBlockType.CLASS:
                            references = [self._fix_reference_path(reference) for reference in self.get_all_references(
                                exclude_types=[CodeBlockType.FUNCTION, CodeBlockType.TEST_CASE]) if
                                          reference and reference.scope != ReferenceScope.EXTERNAL]  # FIXME skip _fix_reference_path?
                        else:
                            references = [self._fix_reference_path(reference) for reference in self.get_all_references() if
                                          reference and reference.scope != ReferenceScope.EXTERNAL]  # FIXME skip _fix_reference_path?

                        for ref in references:
                            path_tree.add_to_tree(ref.path)

                path_tree.add_to_tree(path)
            elif block_path == "":
                path_tree.show = True

        return path_tree


    def to_tree(self,
                indent: int = 0,
                only_identifiers: bool = False,
                show_full_path: bool = True,
                show_tokens: bool = False,
                debug: bool = False,
                include_line_numbers: bool = False,
                include_types: List[CodeBlockType] = None,
                include_parameters: bool = False,
                include_block_delimiters: bool = False,
                include_references: bool = False,
                include_merge_history: bool = False,
                color: bool = False):

        if not include_merge_history and self.type == CodeBlockType.BLOCK_DELIMITER:
            return ""

        child_tree = "".join([
            child.to_tree(indent=indent + 1,
                          only_identifiers=only_identifiers,
                          show_full_path=show_full_path,
                          debug=debug,
                          include_types=include_types,
                          include_line_numbers=include_line_numbers,
                          include_merge_history=include_merge_history,
                          include_parameters=include_parameters,
                          include_references=include_references,
                          include_block_delimiters=include_block_delimiters,
                          show_tokens=show_tokens)
            for child in self.children if not include_types or child.type in include_types])
        indent_str = " " * indent

        extra = ""
        if show_tokens:
            extra += f" ({self.tokens} tokens)"

        if include_references and self.references:
            extra += " references: " + ", ".join([str(ref) for ref in self.references])

        content = Colors.YELLOW + (self.content.strip().replace("\n", "\\n") or "") + Colors.RESET

        if self.identifier:
            if only_identifiers:
                content = ""
            content += Colors.GREEN
            if include_parameters and self.parameters:
                content += f"{self.identifier}({', '.join([param.identifier for param in self.parameters])})"
            elif show_full_path:
                content += f" ({self.path_string()})"
            else:
                content += f" ({self.identifier})"

            content += Colors.RESET

        if include_line_numbers:
            extra += f" {self.start_line}-{self.end_line}"

        if debug and self.properties:
            extra += f" properties: {self.properties}"

        if include_merge_history and self.merge_history:
            extra += " merge_history: " + ", ".join([str(action) for action in self.merge_history])

        return f"{indent_str} {indent} {Colors.BLUE}{self.type.value}{Colors.RESET} `{content}`{extra}\n{child_tree}"

    def __eq__(self, other):
        if not isinstance(other, CodeBlock):
            return False

        return self.full_path() == other.full_path()

    def find_type_in_parents(self, block_type: CodeBlockType) -> Optional["CodeBlock"]:
        if not self.parent:
            return None

        if self.parent.type == block_type:
            return self.parent

        if self.parent:
            return self.parent.find_type_in_parents(block_type)

        return None

    def equal_contents(self, other: "CodeBlock"):
        if len(self.children) != len(other.children):
            return False

        child_equal = all([self.children[i].equal_contents(other.children[i]) for i in range(len(self.children))])
        return self.content == other.content and child_equal

    def dict(self, **kwargs):
        # TODO: Add **kwargs to dict call
        return super().dict(exclude={"parent", "merge_history"})

    def path_string(self):
        return ".".join(self.full_path())

    def full_path(self):
        path = []
        if self.parent:
            path.extend(self.parent.full_path())

        if self.identifier:
            path.append(self.identifier)

        return path

    def root(self):
        if self.parent:
            return self.parent.root()
        return self

    def get_blocks(self, has_identifier: bool, include_types: List[CodeBlockType] = None) -> List["CodeBlock"]:
        blocks = [self]

        for child in self.children:
            if has_identifier and not child.identifier:
                continue

            if include_types and child.type not in include_types:
                continue

            blocks.extend(child.get_indexable_blocks())
        return blocks

    def find_reference(self, ref_path: [str]) -> Optional[Relationship]:
        for child in self.children:
            if child.type == CodeBlockType.IMPORT:
                for reference in child.references:
                    if reference.path[len(reference.path) - len(ref_path):] == ref_path:
                        return reference

            child_path = child.full_path()

            if child_path[len(child_path) - len(ref_path):] == ref_path:
                if self.type == CodeBlockType.CLASS:
                    return Relationship(scope=ReferenceScope.CLASS, path=child_path)
                if self.type == CodeBlockType.MODULE:
                    return Relationship(scope=ReferenceScope.GLOBAL, path=child_path)

                return Relationship(scope=ReferenceScope.LOCAL, path=child_path)

        if self.parent:
            reference = self.parent.find_reference(ref_path)
            if reference:
                return reference

        return None

    def get_all_references(self, exclude_types: List[CodeBlockType] =[]) -> List[Relationship]:
        references = []
        references.extend(self.references)
        for childblock in self.children:
            if not exclude_types or childblock.type not in exclude_types:
                references.extend(childblock.get_all_references(exclude_types=exclude_types))

        return references


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

    def create_commented_out_block(self, comment_out_str: str = "..."):
        return CodeBlock(
            type=CodeBlockType.COMMENTED_OUT_CODE,
            indentation=self.indentation,
            pre_lines=2,
            content=self.create_comment(comment_out_str))

    def create_comment_block(self, comment: str = "..."):
        return CodeBlock(
            type=CodeBlockType.COMMENT,
            indentation=self.indentation,
            pre_lines=1,
            content=self.create_comment(comment))

    def create_comment(self, comment: str) -> str:
        symbol = get_comment_symbol(self.root().language)
        return f"{symbol} {comment}"

    def add_indentation(self, indentation: str):
        if self.pre_lines:
            self.indentation += indentation

        # TODO: Find a more graceful way to solve multi line blocks
        if "\n" in self.content:
            lines = self.content.split("\n")
            content = lines[0]
            for line in lines[1:]:
                if line.startswith(" "):
                    content += "\n" + indentation + line
            self.content = content

        for child in self.children:
            child.add_indentation(indentation)

    def find_equal_parent(self, check_block: "CodeBlock") -> Optional["CodeBlock"]:
        if not self.parent:
            return None

        if self.parent == check_block:
            return self

        return self.parent.find_equal_parent(check_block)

    def _get_matching_content(self, other_children: List["CodeBlock"], start_original: int) -> List["CodeBlock"]:
        original_contents = [child.content for child in self.children[start_original:]]
        return [other_child for other_child in other_children if other_child.content in original_contents]

    def get_matching_blocks(self, other_block: "CodeBlock") -> List["CodeBlock"]:
        matching_children = self._get_matching_content(other_block.children, 0)
        if matching_children:
            return matching_children

        nested_match = self.find_nested_matching_block(other_block)
        if nested_match:
            return [nested_match]

        return []

    def has_any_matching_child(self, other_children: List["CodeBlock"]) -> bool:
        return self._check_matching(other_children, any)

    def has_all_matching_children(self, other_children: List["CodeBlock"]) -> bool:
        if len(self.children) != len(other_children):
            return False
        return self._check_matching(other_children, all)

    def _check_matching(self, other_children: List["CodeBlock"], operation: Callable) -> bool:
        original_identifiers = [child.identifier for child in self.children if child.identifier]
        updated_identifiers = [child.identifier for child in other_children if child.identifier]
        return operation(ub_content in original_identifiers for ub_content in updated_identifiers)

    def find_next_matching_child_block(self, children_start, other_child_block):
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

    def find_by_path(self, path: List[str]) -> Optional["CodeBlock"]:
        if not path:
            return None

        for child in self.children:
            if child.identifier == path[0]:
                if len(path) == 1:
                    return child
                else:
                    return child.find_by_path(path[1:])

        return None

    def has_any_block(self, blocks: List["CodeBlock"]) -> bool:
        for block in blocks:
            if block.full_path()[:len(self.full_path())] == self.full_path():
                return True
        return False

    def find_blocks_with_identifier(self, identifier: str) -> List["CodeBlock"]:
        blocks = []
        for child_block in self.children:
            if child_block.identifier == identifier:
                blocks.append(child_block)
            blocks.extend(child_block.find_blocks_with_identifier(identifier))
        return blocks

    def find_incomplete_blocks_with_type(self, block_type: CodeBlockType):
        return self.find_incomplete_blocks_with_types([block_type])

    def find_incomplete_blocks_with_types(self, block_types: [CodeBlockType]):
        matching_blocks = []
        for child_block in self.children:
            if child_block.type in block_types and not child_block.is_complete():
                matching_blocks.append(child_block)

            if child_block.children:
                matching_blocks.extend(child_block.find_incomplete_blocks_with_types(block_types))

        return matching_blocks

    def find_blocks_with_types(self, block_types: List[CodeBlockType]) -> List["CodeBlock"]:
        matching_blocks = []
        if self.type in block_types:
            matching_blocks.append(self)
        for child_block in self.children:
            matching_blocks.extend(child_block.find_blocks_with_types(block_types=block_types))
        return matching_blocks

    def find_blocks_with_type(self, block_type: CodeBlockType) -> List["CodeBlock"]:
        return self.find_blocks_with_types([block_type])


    def find_indexed_blocks_by_spans(self, spans: List[Span]):
        if self.is_block_within_spans(spans):
            return [self]

        if self.is_spans_within_block(spans):
            if not self.get_indexed_blocks():
                return [self]

            found_blocks = []
            for child in self.children:
                # TODO: Filter out relevant spans
                found_blocks.extend(child.find_indexed_blocks_by_spans(spans))

            return found_blocks

        else:
            return []

    def is_spans_within_block(self, spans: List[Span]) -> bool:
        for span in spans:
            if span.start_line >= self.start_line and span.end_line <= self.end_line:
                return True
        return False

    def is_block_within_spans(self, spans: List[Span]) -> bool:
        for span in spans:
            if span.start_line <= self.start_line and span.end_line >= self.end_line:
                return True
        return False

    def get_indexed_blocks(self) -> List["CodeBlock"]:
        blocks = []
        for child in self.children:
            if child.is_indexed:
                blocks.append(child)

            blocks.extend(child.get_indexed_blocks())

        return blocks
