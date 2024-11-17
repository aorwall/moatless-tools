import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from typing_extensions import deprecated

from moatless.codeblocks.parser.comment import get_comment_symbol
from moatless.utils.colors import Colors

BlockPath = list[str]


class SpanMarker(Enum):
    TAG = 1
    COMMENT = 2


class CodeBlockTypeGroup(str, Enum):
    STRUCTURE = "Structures"
    IMPLEMENTATION = "Implementation"
    IMPORT = "Imports"

    BLOCK_DELIMITER = "BlockDelimiter"
    SPACE = "Space"

    COMMENT = "Comment"

    ERROR = "Error"

    def __str__(self):
        return self.value


class CodeBlockType(Enum):
    MODULE = (
        "Module",
        CodeBlockTypeGroup.STRUCTURE,
    )  # TODO: Module shouldn't be a STRUCTURE
    CLASS = ("Class", CodeBlockTypeGroup.STRUCTURE)
    FUNCTION = ("Function", CodeBlockTypeGroup.STRUCTURE)

    # TODO: Remove and add sub types to functions and classes
    CONSTRUCTOR = ("Constructor", CodeBlockTypeGroup.STRUCTURE)
    TEST_SUITE = ("TestSuite", CodeBlockTypeGroup.STRUCTURE)
    TEST_CASE = ("TestCase", CodeBlockTypeGroup.STRUCTURE)

    IMPORT = ("Import", CodeBlockTypeGroup.IMPORT)

    EXPORT = ("Export", CodeBlockTypeGroup.IMPLEMENTATION)
    COMPOUND = ("Compound", CodeBlockTypeGroup.IMPLEMENTATION)
    # Dependent clauses are clauses that are dependent on another compound statement and can't be shown on their own
    DEPENDENT_CLAUSE = ("DependentClause", CodeBlockTypeGroup.IMPLEMENTATION)
    ASSIGNMENT = ("Assignment", CodeBlockTypeGroup.IMPLEMENTATION)
    CALL = ("Call", CodeBlockTypeGroup.IMPLEMENTATION)
    STATEMENT = ("Statement", CodeBlockTypeGroup.IMPLEMENTATION)

    CODE = ("Code", CodeBlockTypeGroup.IMPLEMENTATION)

    # TODO: Incorporate in code block?
    BLOCK_DELIMITER = ("BlockDelimiter", CodeBlockTypeGroup.BLOCK_DELIMITER)

    # TODO: Remove as it's just to fill upp spaces at the end of the file?
    SPACE = ("Space", CodeBlockTypeGroup.SPACE)

    COMMENT = ("Comment", CodeBlockTypeGroup.COMMENT)
    COMMENTED_OUT_CODE = (
        "Placeholder",
        CodeBlockTypeGroup.COMMENT,
    )  # TODO: Replace to PlaceholderComment

    ERROR = ("Error", CodeBlockTypeGroup.ERROR)

    def __init__(self, value: str, group: CodeBlockTypeGroup):
        self._value = value
        self.group = group

    @property
    def display_name(self):
        return self._value

    def __str__(self):
        return self._value

    @classmethod
    def from_string(cls, tag: str) -> Optional["CodeBlockType"]:
        if not tag.startswith("definition"):
            return None

        tag_to_block_type = {
            "definition.assignment": cls.ASSIGNMENT,
            "definition.block_delimiter": cls.BLOCK_DELIMITER,
            "definition.call": cls.CALL,
            "definition.class": cls.CLASS,
            "definition.code": cls.CODE,
            "definition.comment": cls.COMMENT,
            "definition.compound": cls.COMPOUND,
            "definition.constructor": cls.CONSTRUCTOR,
            "definition.dependent_clause": cls.DEPENDENT_CLAUSE,
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
    CodeBlockType.SPACE,
]

INDEXED_BLOCKS = [
    CodeBlockType.FUNCTION,
    CodeBlockType.CLASS,
    CodeBlockType.TEST_SUITE,
    CodeBlockType.TEST_CASE,
]


@deprecated("Use BlockSpans to define code block visibility instead")
@dataclass
class PathTree:
    show: bool = False
    tree: dict[str, "PathTree"] = field(default_factory=dict)

    @staticmethod
    def from_block_paths(block_paths: list[BlockPath]) -> "PathTree":
        tree = PathTree()
        for block_path in block_paths:
            tree.add_to_tree(block_path)

        return tree

    def child_tree(self, key: str) -> Optional["PathTree"]:
        return self.tree.get(key, None)

    def merge(self, other: "PathTree"):
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


@dataclass
class Relationship:
    scope: ReferenceScope
    external_path: list[str] = field(default_factory=list)
    resolved_path: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    type: RelationshipType = RelationshipType.USES
    identifier: Optional[str] = None

    def __post_init__(self):
        if not self.external_path and not self.path:
            raise ValueError("Cannot create Reference without external_path or path.")

    def __hash__(self):
        return hash((self.scope, tuple(self.path)))

    def __eq__(self, other):
        return (self.scope, self.path) == (other.scope, other.path)

    def full_path(self):
        return self.external_path + self.path

    def __str__(self):
        start_node = self.identifier if self.identifier else ""

        end_node = ""
        if self.external_path:
            end_node = "/".join(self.external_path)
        if self.path:
            if self.external_path:
                end_node += "/"
            end_node += ".".join(self.path)

        return f"({start_node})-[:{self.type.name} {{scope: {self.scope.value}}}]->({end_node})"


@dataclass
class Parameter:
    identifier: str
    type: Optional[str] = None


class SpanType(str, Enum):
    INITATION = "init"
    DOCUMENTATION = "docs"
    IMPLEMENTATION = "impl"


@dataclass
class BlockSpan:
    span_id: str
    span_type: SpanType
    start_line: int
    end_line: int
    block_paths: list[BlockPath] = field(default_factory=list)
    initiating_block: Optional["CodeBlock"] = None
    visible: bool = True
    index: int = 0
    parent_block_path: Optional[BlockPath] = None
    is_partial: bool = False
    tokens: int = 0

    @property
    def block_type(self):
        return self.initiating_block.type if self.initiating_block else None

    def __str__(self):
        return f"{self.span_id} ({self.span_type.value}, {self.tokens} tokens)"

    def get_first_child_block_path(self):
        for block_path in self.block_paths:
            if len(block_path) == len(self.parent_block_path):
                continue
            return block_path


@dataclass
class ValidationError:
    error: str


@dataclass(eq=False, repr=False, slots=True)
class CodeBlock:
    type: CodeBlockType
    content: str
    identifier: Optional[str] = None
    parameters: List["Parameter"] = field(default_factory=list)
    relationships: List["Relationship"] = field(default_factory=list)
    span_ids: Set[str] = field(default_factory=set)
    belongs_to_span: Optional["BlockSpan"] = None
    has_error: bool = False
    start_line: int = 0
    end_line: int = 0
    properties: Dict = field(default_factory=dict)
    pre_code: str = ""
    pre_lines: int = 0
    indentation: str = ""
    tokens: int = 0
    children: List["CodeBlock"] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    parent: Optional["CodeBlock"] = None
    previous: Optional["CodeBlock"] = None
    next: Optional["CodeBlock"] = None

    _content_lines: Optional[List[str]] = field(default=None, init=False)

    def __post_init__(self):
        self._content_lines = None

        if self.children:
            for child in self.children:
                child.parent = self

        if self.pre_code and not self.indentation and not self.pre_lines:
            pre_code_lines = self.pre_code.split("\n")
            self.pre_lines = len(pre_code_lines) - 1
            self.indentation = (
                pre_code_lines[-1] if self.pre_lines > 0 else self.pre_code
            )

    @property
    def content_lines(self):
        if self._content_lines is None:
            self._content_lines = self.content.split("\n")
        return self._content_lines

    def validate_pre_code(self):
        if self.pre_code and not re.match(r"^[ \n\\]*$", self.pre_code):
            raise ValueError(
                f"Failed to parse code block with type {self.type} and content `{self.content}`. "
                f"Expected pre_code to only contain spaces and line breaks. Got `{self.pre_code}`"
            )

    def last(self):
        if self.next:
            return self.next.last()
        return self

    def insert_child(self, index: int, child: "CodeBlock"):
        if index == 0 and self.children[0].pre_lines == 0:
            self.children[0].pre_lines = 1

        self.children.insert(index, child)
        child.parent = self

    def insert_children(self, index: int, children: list["CodeBlock"]):
        for child in children:
            self.insert_child(index, child)
            index += 1

    def append_child(self, child: "CodeBlock"):
        self.children.append(child)
        self.span_ids.update(child.span_ids)
        child.parent = self

    def append_children(self, children: list["CodeBlock"]):
        for child in children:
            self.append_child(child)

    def replace_children(
        self, start_index: int, end_index: int, children: list["CodeBlock"]
    ):
        self.children = (
            self.children[:start_index] + children + self.children[end_index:]
        )
        for child in children:
            child.parent = self

    def replace_child(self, index: int, child: "CodeBlock"):
        # TODO: Do a proper update of everything when replacing child blocks
        child.pre_code = self.children[index].pre_code
        child.pre_lines = self.children[index].pre_lines
        self.sync_indentation(self.children[index], child)

        self.children[index] = child
        child.parent = self

    def remove_child(self, index: int):
        del self.children[index]

    def sync_indentation(self, original_block: "CodeBlock", updated_block: "CodeBlock"):
        original_indentation_length = len(original_block.indentation) + len(
            self.indentation
        )
        updated_indentation_length = len(updated_block.indentation) + len(
            updated_block.parent.indentation
        )

        # To handle separate code blocks provdided out of context
        if (
            original_indentation_length == updated_indentation_length
            and len(updated_block.indentation) == 0
        ):
            updated_block.indentation = " " * original_indentation_length

        elif original_indentation_length > updated_indentation_length:
            additional_indentation = " " * (
                original_indentation_length - updated_indentation_length
            )
            updated_block.add_indentation(additional_indentation)

    def replace_by_path(self, path: list[str], new_block: "CodeBlock"):
        if not path:
            return

        for i, child in enumerate(self.children):
            if child.identifier == path[0]:
                if len(path) == 1:
                    self.replace_child(i, new_block)
                    return
                else:
                    child.replace_by_path(path[1:], new_block)

    def __str__(self):
        return f"{self.display_name} ({self.type.display_name} {self.start_line} - {self.end_line})"

    def to_string(self):
        return self._to_string()

    def sum_tokens(self):
        tokens = self.tokens
        tokens += sum([child.sum_tokens() for child in self.children])
        return tokens

    def get_all_child_blocks(self) -> list["CodeBlock"]:
        blocks = []
        for child in self.children:
            blocks.append(child)
            blocks.extend(child.get_all_child_blocks())
        return blocks

    def get_children(
        self, exclude_blocks: list[CodeBlockType] = None
    ) -> list["CodeBlock"]:
        if exclude_blocks is None:
            exclude_blocks = []
        return [child for child in self.children if child.type not in exclude_blocks]

    def show_related_spans(
        self,
        span_id: Optional[str] = None,  # TODO: Set max tokens to show
    ):
        related_spans = self.find_related_spans(span_id)
        for span in related_spans:
            span.visible = True

    def has_visible_children(self):
        for child in self.children:
            if child.is_visible:
                return True

            if child.has_visible_children():
                return True

        return False

    @property
    def is_visible(self):
        return self.belongs_to_span and self.belongs_to_span.visible

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

        for _i, child in enumerate(self.children):
            contents += child._to_string()

        return contents

    def _build_path_tree(
        self, block_paths: list[str], include_references: bool = False
    ):
        path_tree = PathTree()

        for block_path in block_paths:
            if block_path:
                path = block_path.split(".")
                if include_references:
                    block = self.find_by_path(path)
                    if block:
                        if self.type == CodeBlockType.CLASS:
                            references = [
                                self._fix_reference_path(reference)
                                for reference in self.get_all_relationships(
                                    exclude_types=[
                                        CodeBlockType.FUNCTION,
                                        CodeBlockType.TEST_CASE,
                                    ]
                                )
                                if reference
                                and reference.scope != ReferenceScope.EXTERNAL
                            ]  # FIXME skip _fix_reference_path?
                        else:
                            references = [
                                self._fix_reference_path(reference)
                                for reference in self.get_all_relationships()
                                if reference
                                and reference.scope != ReferenceScope.EXTERNAL
                            ]  # FIXME skip _fix_reference_path?

                        for ref in references:
                            path_tree.add_to_tree(ref.path)

                path_tree.add_to_tree(path)
            elif block_path == "":
                path_tree.show = True

        return path_tree

    def to_tree(
        self,
        indent: int = 0,
        current_span: BlockSpan | None = None,
        highlight_spans: set[str] | None = None,
        only_identifiers: bool = False,
        show_full_path: bool = True,
        show_tokens: bool = False,
        show_spans: bool = False,
        debug: bool = False,
        exclude_not_highlighted: bool = False,
        include_line_numbers: bool = False,
        include_types: list[CodeBlockType] | None = None,
        include_parameters: bool = False,
        include_block_delimiters: bool = False,
        include_references: bool = False,
        include_merge_history: bool = False,
    ):
        if not include_merge_history and self.type == CodeBlockType.BLOCK_DELIMITER:
            return ""

        indent_str = " " * indent

        highlighted = False

        child_tree = ""
        for _i, child in enumerate(self.children):
            if child.belongs_to_span and (
                not current_span
                or current_span.span_id != child.belongs_to_span.span_id
            ):
                current_span = child.belongs_to_span

                highlighted = highlight_spans is None or (
                    current_span is not None and current_span.span_id in highlight_spans
                )

                if show_spans:
                    color = Colors.WHITE if highlighted else Colors.GRAY
                    child_tree += f"{indent_str} {indent} {color}Span: {current_span}{Colors.RESET}\n"

            if (
                exclude_not_highlighted
                and not highlighted
                and not child.has_any_span(highlight_spans)
            ):
                continue

            child_tree += child.to_tree(
                indent=indent + 1,
                current_span=current_span,
                highlight_spans=highlight_spans,
                exclude_not_highlighted=exclude_not_highlighted,
                only_identifiers=only_identifiers,
                show_full_path=show_full_path,
                show_tokens=show_tokens,
                debug=debug,
                show_spans=show_spans,
                include_line_numbers=include_line_numbers,
                include_types=include_types,
                include_parameters=include_parameters,
                include_block_delimiters=include_block_delimiters,
                include_references=include_references,
                include_merge_history=include_merge_history,
            )

        is_visible = not highlight_spans or self.belongs_to_any_span(highlight_spans)
        extra = ""
        if show_tokens:
            extra += f" ({self.tokens} tokens)"

        if include_references and self.relationships:
            extra += " references: " + ", ".join(
                [str(ref) for ref in self.relationships]
            )

        content = (
            Colors.YELLOW
            if is_visible
            else Colors.GRAY
            + (self.content.strip().replace("\n", "\\n") or "")
            + Colors.RESET
        )

        if self.identifier:
            if only_identifiers:
                content = ""
            content += Colors.GREEN if is_visible else Colors.GRAY
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
            extra += " merge_history: " + ", ".join(
                [str(action) for action in self.merge_history]
            )

        type_color = Colors.BLUE if is_visible else Colors.GRAY
        return f"{indent_str} {indent} {type_color}{self.type.value}{Colors.RESET} `{content}`{extra}{Colors.RESET}\n{child_tree}"

    def _to_prompt_string(
        self,
        show_span_id: bool = False,
        span_marker: SpanMarker = SpanMarker.COMMENT,
        show_line_numbers: bool = False,
    ) -> str:
        contents = ""

        if show_span_id:
            contents += "\n\n"
            if span_marker == SpanMarker.COMMENT:
                span_comment = self.create_comment(
                    f"span_id: {self.belongs_to_span.span_id}"
                )
                contents += f"{self.indentation}{span_comment}"
            elif span_marker == SpanMarker.TAG:
                contents += f"\n<span id='{self.belongs_to_span.span_id}'>"

            if not self.pre_lines:
                contents += "\n"

        def print_line(line_number: int):
            if not show_line_numbers:
                return ""

            # Don't print out line numbers on out commented code to make it harder for the LLM to select it
            if (
                line_number == self.start_line
                and self.type == CodeBlockType.COMMENTED_OUT_CODE
            ):
                return " " * 6
            return f"{line_number:6}\t"

        # Just to write out the first line number when there are no pre_lines on first block
        if (
            not self.pre_lines
            and self.parent
            and self.parent.type == CodeBlockType.MODULE
            and self.parent.children[0] == self
        ):
            contents += print_line(self.start_line)

        if self.pre_lines:
            for i in range(self.pre_lines):
                contents += "\n"
                contents += print_line(self.start_line - self.pre_lines + i + 1)

        contents += self.indentation + self.content_lines[0]
        for i, line in enumerate(self.content_lines[1:]):
            contents += "\n"
            contents += print_line(self.start_line + i + 1)
            contents += line

        return contents

    def to_prompt(
        self,
        span_ids: set[str] | None = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        show_outcommented_code: bool = True,
        outcomment_code_comment: str = "...",
        show_span_id: bool = False,
        current_span_id: Optional[str] = None,
        show_line_numbers: bool = False,
        exclude_block_types: list[CodeBlockType] | None = None,
        include_block_types: list[CodeBlockType] | None = None,
    ):
        contents = ""
        show_new_span_id = (
            show_span_id
            and self.belongs_to_span
            and (not current_span_id or current_span_id != self.belongs_to_span.span_id)
        )
        contents += self._to_prompt_string(
            show_span_id=show_new_span_id, show_line_numbers=show_line_numbers
        )

        has_outcommented_code = False
        for _i, child in enumerate(self.children):
            show_child = True

            if exclude_block_types and child.type in exclude_block_types:
                show_child = False

            if show_child and span_ids:
                show_child = child.has_any_span(span_ids)

            if show_child and include_block_types:
                show_child = child.has_blocks_with_types(include_block_types)

            if show_child and start_line and end_line:
                show_child = child.has_lines(
                    start_line, end_line
                ) or child.is_within_lines(start_line, end_line)

            if show_child:
                if has_outcommented_code:
                    contents += child.create_commented_out_block(
                        outcomment_code_comment
                    ).to_string()

                has_outcommented_code = False

                if child.belongs_to_span:
                    current_span_id = child.belongs_to_span.span_id

                contents += child.to_prompt(
                    span_ids=span_ids,
                    start_line=start_line,
                    end_line=end_line,
                    show_outcommented_code=show_outcommented_code,
                    outcomment_code_comment=outcomment_code_comment,
                    show_span_id=show_span_id,
                    current_span_id=current_span_id,
                    show_line_numbers=show_line_numbers,
                    exclude_block_types=exclude_block_types,
                    include_block_types=include_block_types,
                )
            elif show_outcommented_code and child.type not in [
                CodeBlockType.COMMENT,
                CodeBlockType.COMMENTED_OUT_CODE,
            ]:
                has_outcommented_code = True

        if (
            outcomment_code_comment
            and has_outcommented_code
            and child.type
            not in [
                CodeBlockType.COMMENT,
                CodeBlockType.COMMENTED_OUT_CODE,
                CodeBlockType.SPACE,
            ]
        ):
            contents += "\n.    " if show_line_numbers else "\n"
            contents += child.create_commented_out_block(
                outcomment_code_comment
            ).to_string()
            contents += "\n"

        return contents

    def __eq__(self, other):
        if not isinstance(other, CodeBlock):
            return False

        return self.full_path() == other.full_path()

    def compare_indentation(self, other_block: "CodeBlock"):
        existing_indentation = len(self.indentation)
        new_indentation = len(other_block.indentation)
        return existing_indentation - new_indentation

    def find_block_by_type(self, block_type: CodeBlockType) -> Optional["CodeBlock"]:
        if self.type == block_type:
            return self

        for child in self.children:
            block = child.find_block_by_type(block_type)
            if block:
                return block

        return None

    def find_type_in_parents(self, block_type: CodeBlockType) -> Optional["CodeBlock"]:
        if not self.parent:
            return None

        if self.parent.type == block_type:
            return self.parent

        if self.parent:
            return self.parent.find_type_in_parents(block_type)

        return None

    def structure_block(self):
        if self.type.group == CodeBlockTypeGroup.STRUCTURE:
            return self

        if self.parent:
            return self.parent.structure_block()

        return None

    def find_type_group_in_parents(
        self, block_type_group: CodeBlockTypeGroup
    ) -> Optional["CodeBlock"]:
        if not self.parent:
            return None

        if self.parent.type.group == block_type_group:
            return self.parent

        if self.parent:
            return self.parent.find_type_group_in_parents(block_type_group)

        return None

    def find_spans_by_line_numbers(
        self, start_line: int, end_line: int | None = None
    ) -> list[BlockSpan]:
        spans = []
        for child in self.children:
            if end_line is None:
                end_line = start_line

            if child.end_line < start_line:
                continue

            if child.start_line > end_line:
                if not spans:
                    last_block = self.find_last_by_end_line(end_line)
                    if last_block:
                        spans.append(last_block.belongs_to_span)
                return spans

            if (
                child.belongs_to_span
                and child.belongs_to_span.span_id not in spans
                and (
                    not child.children
                    or child.children[0].start_line > end_line
                    or (child.start_line >= start_line and child.end_line <= end_line)
                    or child.start_line == start_line
                    or child.end_line == end_line
                )
            ):
                spans.append(child.belongs_to_span)

            child_spans = child.find_spans_by_line_numbers(start_line, end_line)
            for span in child_spans:
                if span not in spans:
                    spans.append(span)

        return spans

    def dict(self, **kwargs):
        # TODO: Add **kwargs to dict call
        return super().dict(exclude={"parent", "merge_history"})

    @property
    def display_name(self):
        if self.full_path():
            return self.path_string()
        else:
            return "<module>"

    def path_string(self):
        return ".".join(self.full_path())

    def full_path(self):
        path = []
        if self.parent:
            path.extend(self.parent.full_path())

        if self.identifier:
            path.append(self.identifier)

        return path

    @property
    def module(self) -> "Module":  # noqa: F821
        if self.parent:
            return self.parent.module
        return None

    @deprecated("Use codeblock.module")
    def root(self) -> "Module":  # noqa: F821
        return self.module

    def get_blocks(
        self, has_identifier: bool, include_types: list[CodeBlockType] | None = None
    ) -> list["CodeBlock"]:
        blocks = [self]

        for child in self.children:
            if has_identifier and not child.identifier:
                continue

            if include_types and child.type not in include_types:
                continue

            blocks.extend(child.get_indexable_blocks())
        return blocks

    def find_reference(self, ref_path: [str]) -> Relationship | None:
        for child in self.children:
            if child.type == CodeBlockType.IMPORT:
                for reference in child.relationships:
                    if (
                        reference.path[len(reference.path) - len(ref_path) :]
                        == ref_path
                    ):
                        return reference

            child_path = child.full_path()

            if child_path[len(child_path) - len(ref_path) :] == ref_path:
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

    def get_all_relationships(
        self, exclude_types: list[CodeBlockType] = None
    ) -> list[Relationship]:
        if exclude_types is None:
            exclude_types = []
        references = []
        references.extend(self.relationships)
        for childblock in self.children:
            if not exclude_types or childblock.type not in exclude_types:
                references.extend(
                    childblock.get_all_relationships(exclude_types=exclude_types)
                )

        return references

    def is_complete(self):
        if self.type == CodeBlockType.COMMENTED_OUT_CODE:
            return False
        return all(child.is_complete() for child in self.children)

    def find_errors(self) -> list[str]:
        errors = []

        if self.children:
            for child in self.children:
                errors.extend(child.find_errors())

        if self.type == CodeBlockType.ERROR:
            if self.validation_errors:
                errors.extend(self.validation_errors)
            else:
                errors.append(f"Found validation errors in {self.path_string()}")

        return errors

    def create_commented_out_block(self, comment_out_str: str = "..."):
        return CodeBlock(
            type=CodeBlockType.COMMENTED_OUT_CODE,
            indentation=self.indentation,
            parent=self,
            pre_lines=1,
            content=self.create_comment(comment_out_str),
        )

    def create_comment_block(self, comment: str = "...", pre_lines: int = 1):
        return CodeBlock(
            type=CodeBlockType.COMMENT,
            indentation=self.indentation,
            parent=self,
            pre_lines=pre_lines,
            content=self.create_comment(comment),
        )

    def create_comment(self, comment: str) -> str:
        symbol = get_comment_symbol("python")  # FIXME: Derive language from Module
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

    def find_by_path(self, path: list[str]) -> Optional["CodeBlock"]:
        if path is None:
            return None

        if not path:
            return self

        for child in self.children:
            if child.identifier == path[0]:
                if len(path) == 1:
                    return child
                else:
                    return child.find_by_path(path[1:])

        return None

    def find_blocks_by_span_id(self, span_id: str) -> list["CodeBlock"]:
        blocks = []
        if self.belongs_to_span and self.belongs_to_span.span_id == span_id:
            blocks.append(self)

        for child in self.children:
            # TODO: Optimize to just check relevant children (by mapping spans?
            blocks.extend(child.find_blocks_by_span_id(span_id))

        return blocks

    def find_last_before_span(
        self, span_id: str, last_before_span: Optional["CodeBlock"] = None
    ) -> Optional["CodeBlock"]:
        if self.belongs_to_span and self.belongs_to_span.span_id == span_id:
            return last_before_span

        for child in self.children:
            if child.belongs_to_span and child.belongs_to_span.span_id == span_id:
                return last_before_span

            if child.belongs_to_span and child.belongs_to_span.span_id != span_id:
                last_before_span = child

            result = child.find_last_before_span(span_id, last_before_span)
            if result:
                return result

        return None

    def find_first_by_span_id(self, span_id: str) -> Optional["CodeBlock"]:
        if self.belongs_to_span and self.belongs_to_span.span_id == span_id:
            return self

        for child in self.children:
            found = child.find_first_by_span_id(span_id)
            if found:
                return found

        return None

    def find_last_by_span_id(self, span_id: str) -> Optional["CodeBlock"]:
        for child in reversed(self.children):
            if child.belongs_to_span and child.belongs_to_span.span_id == span_id:
                return child

            found = child.find_last_by_span_id(span_id)
            if found:
                return found

        return None

    def has_any_block(self, blocks: list["CodeBlock"]) -> bool:
        for block in blocks:
            if block.full_path()[: len(self.full_path())] == self.full_path():
                return True
        return False

    def find_by_identifier(
        self,
        identifier: str,
        type: CodeBlockType | None = None,
        recursive: bool = False,
    ):
        for child in self.children:
            if child.identifier == identifier and (not type or child.type == type):
                return child

            if recursive:
                found = child.find_by_identifier(identifier, type, recursive)
                if found:
                    return found
        return None

    def find_blocks_with_identifier(self, identifier: str) -> list["CodeBlock"]:
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
                matching_blocks.extend(
                    child_block.find_incomplete_blocks_with_types(block_types)
                )

        return matching_blocks

    def find_blocks_with_types(
        self, block_types: list[CodeBlockType]
    ) -> list["CodeBlock"]:
        matching_blocks = []
        if self.type in block_types:
            matching_blocks.append(self)
        for child_block in self.children:
            matching_blocks.extend(
                child_block.find_blocks_with_types(block_types=block_types)
            )
        return matching_blocks

    def has_blocks_with_types(self, block_types: list[CodeBlockType]) -> bool:
        if self.type in block_types:
            return True
        for child_block in self.children:
            if child_block.has_blocks_with_types(block_types):
                return True
        return False

    def has_placeholders(self):
        return self.find_blocks_with_type(CodeBlockType.COMMENTED_OUT_CODE)

    def find_blocks_with_type(self, block_type: CodeBlockType) -> list["CodeBlock"]:
        return self.find_blocks_with_types([block_type])

    def find_first_by_start_line(self, start_line: int) -> Optional["CodeBlock"]:
        for child in self.children:
            if child.start_line >= start_line:
                return child

            if child.end_line >= start_line:
                if not child.children:
                    return child

                found = child.find_first_by_start_line(start_line)
                if found:
                    return found

        return None

    def find_blocks_by_line_numbers(
        self,
        start_line: int,
        end_line: int | None = None,
        include_parents: bool = False,
    ) -> List["CodeBlock"]:
        blocks = []
        block = self
        while block.next and (end_line is None or block.start_line <= end_line):
            if include_parents and block.has_lines(start_line, end_line):
                blocks.append(block)
            elif block.start_line >= start_line:
                blocks.append(block)
            block = block.next

        return blocks

    def find_last_by_end_line(
        self, end_line: int, tokens: Optional[int] = None
    ) -> Optional["CodeBlock"]:
        last_child = None
        for child in self.children:
            if child.start_line > end_line or (tokens and child.tokens > tokens):
                return last_child

            if tokens:
                tokens -= child.tokens

            last_child = child

            if child.end_line > end_line:
                found = child.find_last_by_end_line(end_line, tokens=tokens)
                if found:
                    return found

        return None

    def line_within_token_context(self, line_number: int, tokens: int) -> bool:
        if tokens <= 0:
            return False

        if self.end_line < line_number:
            if not self.next:
                return False
            if self.next.start_line > line_number:
                return True
            else:
                return self.next.line_within_token_context(
                    line_number, tokens - self.tokens
                )
        else:
            if not self.previous:
                return False
            elif self.previous.end_line < line_number:
                return True
            else:
                return self.previous.line_within_token_context(
                    line_number, tokens - self.tokens
                )

    def find_last_previous_block_with_block_group(
        self, block_group: CodeBlockTypeGroup
    ):
        if not self.previous:
            return None

        if self.previous.type.group == block_group:
            return self.previous

        return self.previous.find_last_previous_block_with_block_group(block_group)

    def find_next_block_with_block_group(self, block_group: CodeBlockTypeGroup):
        if not self.next:
            return None

        if self.next.type.group == block_group:
            return self.next

        return self.next.find_next_block_with_block_group(block_group)

    def tokens_from_line(self, line_number: int) -> Optional[int]:
        if not self.previous or self.previous.end_line < line_number:
            return self.tokens

        return self.tokens + self.previous.tokens_from_line(line_number)

    def last_block_until_line(self, line_number: int, tokens: int) -> "CodeBlock":
        if self.end_line < line_number:
            if (
                not self.next
                or self.next.start_line > line_number
                or self.next.tokens > tokens
            ):
                return self
            else:
                return self.next.last_block_until_line(
                    line_number, tokens - self.tokens
                )
        else:
            if (
                not self.previous
                or self.previous.end_line < line_number
                or self.next.tokens > tokens
            ):
                return self
            else:
                return self.previous.last_block_until_line(
                    line_number, tokens - self.tokens
                )

    def get_all_span_ids(self, include_self: bool = True) -> set[str]:
        span_ids = set()

        if include_self and self.belongs_to_span:
            span_ids.add(self.belongs_to_span.span_id)

        for child in self.children:
            span_ids.update(child.get_all_span_ids())

        return span_ids

    def get_all_spans(self, include_self: bool = True) -> list[BlockSpan]:
        span_ids = self.get_all_span_ids(include_self=include_self)
        return [self.module.find_span_by_id(span_id) for span_id in span_ids]

    def has_span(self, span_id: str):
        return self.has_any_span({span_id})

    def has_any_span(self, span_ids: set[str]):
        all_span_ids = self.get_all_span_ids(include_self=False)
        return any([span_id in all_span_ids for span_id in span_ids])

    def belongs_to_any_span(self, span_ids: set[str]):
        return self.belongs_to_span and self.belongs_to_span.span_id in span_ids

    def has_lines(self, start_line: int, end_line: int | None = None):
        # Returns True if any part of the block is within the provided line range
        if end_line is None:
            return self.end_line >= start_line
        return not (self.end_line < start_line or self.start_line > end_line)

    def is_within_lines(self, start_line: int, end_line: int):
        return self.start_line >= start_line and self.end_line <= end_line

    def has_content(self, query: str, span_id: Optional[str] = None):
        if (
            self.content
            and query in self.content
            and (
                not span_id
                or (self.belongs_to_span and self.belongs_to_span.span_id == span_id)
            )
        ):
            return True

        if span_id and not self.has_span(span_id):
            return False

        return any(child.has_content(query, span_id) for child in self.children)
