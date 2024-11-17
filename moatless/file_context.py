import difflib
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Set

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from unidiff import PatchSet

from moatless.codeblocks import CodeBlockType, get_parser_by_path
from moatless.codeblocks.codeblocks import (
    BlockSpan,
    CodeBlock,
    CodeBlockTypeGroup,
    SpanMarker,
    SpanType,
)
from moatless.codeblocks.module import Module
from moatless.index import CodeIndex
from moatless.repository import FileRepository
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.schema import FileWithSpans

logger = logging.getLogger(__name__)


class ContextSpan(BaseModel):
    span_id: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    tokens: Optional[int] = None
    pinned: bool = Field(
        default=False,
        description="Whether the span is pinned and cannot be removed from context",
    )


@dataclass
class CurrentPromptSpan:
    span_id: Optional[str] = None
    tokens: int = 0


class ContextFile(BaseModel):
    """
    Represents the context of a file, managing patches that reflect changes over time.

    Attributes:
        file_path (str): The path to the file within the repository.
        accumulated_patch (Optional[str]): A Git-formatted patch representing all changes from the original content.
        patch (Optional[str]): A Git-formatted patch representing the latest changes applied in this ContextFile.
        spans (List[ContextSpan]): A list of spans associated with this file.
        show_all_spans (bool): A flag to indicate whether to display all spans.
    """

    file_path: str = Field(
        ..., description="The relative path to the file within the repository."
    )
    patch: Optional[str] = Field(
        None,
        description="Git-formatted patch representing the latest changes applied in this ContextFile.",
    )
    spans: List[ContextSpan] = Field(
        default_factory=list,
        description="List of context spans associated with this file.",
    )
    show_all_spans: bool = Field(
        False, description="Flag to indicate whether to display all context spans."
    )

    # Private attributes
    _initial_patch: Optional[str] = PrivateAttr(None)
    _cached_base_content: Optional[str] = PrivateAttr(None)
    _cached_content: Optional[str] = PrivateAttr(None)
    _cached_module: Optional[Module] = PrivateAttr(None)

    _repo: Repository = PrivateAttr()

    _cache_valid: bool = PrivateAttr(False)

    def __init__(
        self,
        repo: Repository,
        file_path: str,
        initial_patch: Optional[str] = None,
        **data,
    ):
        """
        Initializes the ContextFile instance.

        Args:
            repo (Repository): The repository instance.
            file_path (str): The path to the file within the repository.
            initial_patch (Optional[str]): A Git-formatted patch representing accumulated changes from the original content.
            **data: Arbitrary keyword arguments. Must include '_repo'.

        Raises:
            ValueError: If '_repo' is not provided in the initialization data.
        """
        super().__init__(file_path=file_path, **data)
        self._repo = repo
        self._initial_patch = initial_patch

    def _add_import_span(self):
        # TODO: Initiate module or add this lazily?
        if self.module:
            # Always include init spans like 'imports' to context file
            for child in self.module.children:
                if (
                    child.type == CodeBlockType.IMPORT
                ) and child.belongs_to_span.span_id:
                    self.add_span(child.belongs_to_span.span_id, pinned=True)

    def get_base_content(self) -> str:
        """
        Retrieves the base content of the file by applying the initial_patch to the original content.

        Returns:
            str: The base content of the file.

        Raises:
            FileNotFoundError: If the file does not exist in the repository.
            Exception: If applying the initial_patch fails.
        """
        if self._cached_base_content is not None:
            return self._cached_base_content

        if not self._repo.file_exists(self.file_path):
            original_content = ""
        else:
            original_content = self._repo.get_file_content(self.file_path)

        if self._initial_patch:
            try:
                self._cached_base_content = self.apply_patch_to_content(
                    original_content, self._initial_patch
                )
            except Exception as e:
                raise Exception(f"Failed to apply initial patch: {e}")
        else:
            self._cached_base_content = original_content

        return self._cached_base_content

    @property
    def module(self) -> Module | None:
        if self._cached_module is not None:
            return self._cached_module

        parser = get_parser_by_path(self.file_path)
        if parser:
            self._cached_module = parser.parse(self.content)

        return self._cached_module

    @property
    def content(self) -> str:
        """
        Retrieves the current content of the file by applying the latest patch to the base content.

        Returns:
            str: The current content of the file.
        """
        if self._cached_content is not None:
            return self._cached_content

        base_content = self.get_base_content()
        if self.patch:
            try:
                self._cached_content = self.apply_patch_to_content(
                    base_content, self.patch
                )
            except Exception as e:
                logger.error(f"Failed to apply patch: {self.patch}")
                raise e
        else:
            self._cached_content = base_content

        return self._cached_content

    def apply_changes(self, updated_content: str) -> set[str]:
        """
        Applies new content to the ContextFile by generating a patch between the base content and the new content.

        Args:
            updated_content (str): The new content to apply to the file.
        """

        if self.module:
            existing_span_ids = self.module.get_all_span_ids()
        else:
            existing_span_ids = set()

        base_content = self.get_base_content()
        new_patch = self.generate_patch(base_content, updated_content)
        self.patch = new_patch

        if self.module:
            updated_span_ids = self.module.get_all_span_ids()
            new_span_ids = updated_span_ids - existing_span_ids
        else:
            new_span_ids = set()

        self.add_spans(new_span_ids)

        # Invalidate cached content
        self._cached_content = None
        self._cached_module = None

        return new_span_ids

    def add_patch(self, new_patch: Optional[str]):
        """
        Accumulates a new patch into the initial_patch and updates the 'patch' field.

        Args:
            new_patch (Optional[str]): The new Git-formatted patch to add.
        """
        if new_patch is None:
            return  # Nothing to add

        if self._initial_patch is None:
            self._initial_patch = new_patch
        else:
            # Combine the existing initial_patch with the new_patch to form a new initial_patch
            base_content = self.get_base_content()
            try:
                # Apply the existing initial_patch to get intermediate content
                intermediate_content = self.apply_patch_to_content(
                    self._repo.get_file_content(self.file_path), self._initial_patch
                )
                # Apply the new_patch to the intermediate content
                updated_content = self.apply_patch_to_content(
                    intermediate_content, new_patch
                )
                # Generate a new initial_patch from original to updated_content
                self._initial_patch = self.generate_patch(
                    self._repo.get_file_content(self.file_path), updated_content
                )
            except Exception as e:
                raise Exception(f"Failed to accumulate patch: {e}")

        # Update the 'patch' field with the latest patch (which could be None or new_patch)
        self.patch = new_patch

        # Invalidate cached content
        self._cached_content = None
        self._cached_module = None
        self._cached_base_content = None

    def apply_patch_to_content(self, content: str, patch: str) -> str:
        """
        Applies a Git-formatted patch to the given content.

        Args:
            content (str): The original content to apply the patch to.
            patch (str): The Git-formatted patch to apply.

        Returns:
            str: The patched content.

        Raises:
            Exception: If the patch does not contain changes for the specified file or if a context mismatch occurs.
        """
        patch_set = PatchSet(io.StringIO(patch))
        patched_content = content

        for patched_file in patch_set:
            patched_file_path = patched_file.path
            # Correctly strip 'a/' or 'b/' prefixes
            if patched_file_path.startswith("a/"):
                patched_file_path = patched_file_path[2:]
            elif patched_file_path.startswith("b/"):
                patched_file_path = patched_file_path[2:]
            if os.path.normpath(patched_file_path) == os.path.normpath(self.file_path):
                patched_content = self._apply_patched_file(
                    patched_content, patched_file
                )
                break
        else:
            raise Exception(f"Patch does not contain changes for file {self.file_path}")

        return patched_content

    def _apply_patched_file(self, content: str, patched_file) -> str:
        """
        Applies a single patched file's hunks to the content.

        Args:
            content (str): The original content.
            patched_file: The patched file object from the PatchSet.

        Returns:
            str: The patched content.

        Raises:
            Exception: If there is a context mismatch during patch application.
        """
        content_lines = content.splitlines(keepends=True)
        new_content_lines = []
        line_no = 0  # 0-based index

        for hunk in patched_file:
            try:
                # Copy unchanged lines before the hunk
                while line_no < hunk.source_start - 1 and line_no < len(content_lines):
                    new_content_lines.append(content_lines[line_no])
                    line_no += 1

                # Apply changes from the hunk
                for line in hunk:
                    if line.is_context:
                        if line_no >= len(content_lines):
                            raise Exception(
                                f"Patch context mismatch: Line {line_no} is beyond end of file"
                            )
                        elif (
                            line.value.strip()
                            and content_lines[line_no].strip() != line.value.strip()
                        ):
                            raise Exception(
                                f'Patch context mismatch at line {line_no}: Expected "{line.value.strip()}", got "{content_lines[line_no].strip()}"'
                            )
                        new_content_lines.append(content_lines[line_no])
                        line_no += 1
                    elif line.is_added:
                        new_content_lines.append(line.value)
                    elif line.is_removed:
                        if line_no >= len(content_lines):
                            raise Exception(
                                f"Patch context mismatch: Cannot remove line {line_no} as it is beyond end of file"
                            )
                        line_no += 1

            except Exception as e:
                raise RuntimeError(
                    f"Failed to apply patch to {self.file_path}. Line number {line_no}. Hunk: {hunk}"
                ) from e

        # Copy any remaining lines after the last hunk
        if line_no < len(content_lines):
            new_content_lines.extend(content_lines[line_no:])

        return "".join(new_content_lines)

    def generate_full_patch(self) -> str:
        """
        Generates a full Git-formatted patch from the original content to the current content.

        Returns:
            str: The generated full patch as a string.
        """
        original_content = self._repo.get_file_content(self.file_path)
        current_content = self.content

        print(f"original_content: {original_content}")
        print(f"current_content: {current_content}")
        patch = self.generate_patch(original_content, current_content)
        print(f"patch: {patch}")
        return patch

    def generate_patch(self, old_content: str, new_content: str) -> str:
        """
        Generates a Git-formatted unified diff patch between old_content and new_content.

        Args:
            old_content (str): The original content.
            new_content (str): The new content.

        Returns:
            str: The generated patch as a string.
        """

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Ensure that new content end with a newline
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="a/" + self.file_path,
                tofile="b/" + self.file_path,
                # lineterm=''
            )
        )

        return "".join(diff_lines)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Ensure 'patch' is always included, even if it's None
        if "patch" not in data:
            data["patch"] = None
        return data

    @property
    def span_ids(self):
        return {span.span_id for span in self.spans}

    def to_prompt(
        self,
        show_span_ids=False,
        show_line_numbers=False,
        exclude_comments=False,
        show_outcommented_code=False,
        outcomment_code_comment: str = "...",
        show_all_spans: bool = False,
        only_signatures: bool = False,
    ):
        if self.module:
            if (
                not self.show_all_spans
                and self.span_ids is not None
                and len(self.span_ids) == 0
            ):
                logger.warning(
                    f"No span ids provided for {self.file_path}, return empty"
                )
                return ""

            code = self._to_prompt(
                code_block=self.module,
                show_span_id=show_span_ids,
                show_line_numbers=show_line_numbers,
                outcomment_code_comment=outcomment_code_comment,
                show_outcommented_code=show_outcommented_code,
                exclude_comments=exclude_comments,
                show_all_spans=show_all_spans or self.show_all_spans,
                only_signatures=only_signatures,
            )
        else:
            code = self._to_prompt_with_line_spans(show_span_id=show_span_ids)

        return f"{self.file_path}\n```\n{code}\n```\n"

    def _find_span(self, codeblock: CodeBlock) -> Optional[ContextSpan]:
        if not codeblock.belongs_to_span:
            return None

        for span in self.spans:
            if codeblock.belongs_to_span.span_id == span.span_id:
                return span

        return None

    def _within_span(self, line_no: int) -> Optional[ContextSpan]:
        for span in self.spans:
            if (
                span.start_line
                and span.end_line
                and span.start_line <= line_no <= span.end_line
            ):
                return span
        return None

    def _to_prompt_with_line_spans(self, show_span_id: bool = False) -> str:
        content_lines = self.content.split("\n")

        if not self.span_ids:
            return self.content

        prompt_content = ""
        outcommented = True
        for i, line in enumerate(content_lines):
            line_no = i + 1

            span = self._within_span(line_no)
            if span:
                if outcommented and show_span_id:
                    prompt_content += f"<span id={span.span_id}>\n"

                prompt_content += line + "\n"
                outcommented = False
            elif not outcommented:
                prompt_content += "... other code\n"
                outcommented = True

        return prompt_content

    def _to_prompt(
        self,
        code_block: CodeBlock,
        current_span: Optional[CurrentPromptSpan] = None,
        show_outcommented_code: bool = True,
        outcomment_code_comment: str = "...",
        show_span_id: bool = False,
        show_line_numbers: bool = False,
        exclude_comments: bool = False,
        show_all_spans: bool = False,
        only_signatures: bool = False,
    ):
        if current_span is None:
            current_span = CurrentPromptSpan()

        contents = ""
        if not code_block.children:
            return contents

        outcommented_block = None
        for _i, child in enumerate(code_block.children):
            if exclude_comments and child.type.group == CodeBlockTypeGroup.COMMENT:
                continue

            show_new_span_id = False
            show_child = False
            child_span = self._find_span(child)

            if child_span:
                if child_span.span_id != current_span.span_id:
                    show_child = True
                    show_new_span_id = show_span_id
                    current_span = CurrentPromptSpan(child_span.span_id)
                elif not child_span.tokens:
                    show_child = True
                else:
                    # Count all tokens in child block if it's not a structure (function or class) or a 'compound' (like an 'if' or 'for' clause)
                    if (
                        child.type.group == CodeBlockTypeGroup.IMPLEMENTATION
                        and child.type
                        not in [CodeBlockType.COMPOUND, CodeBlockType.DEPENDENT_CLAUSE]
                    ):
                        child_tokens = child.sum_tokens()
                    else:
                        child_tokens = child.tokens

                    if current_span.tokens + child_tokens <= child_span.tokens:
                        show_child = True

                    current_span.tokens += child_tokens

            elif (
                not child.belongs_to_span or child.belongs_to_any_span not in self.spans
            ) and child.has_any_span(self.span_ids):
                show_child = True

                if (
                    child.belongs_to_span
                    and current_span.span_id != child.belongs_to_span.span_id
                ):
                    show_new_span_id = show_span_id
                    current_span = CurrentPromptSpan(child.belongs_to_span.span_id)

            if self.show_all_spans or show_all_spans:
                show_child = True

            if only_signatures and child.type.group != CodeBlockTypeGroup.STRUCTURE:
                show_child = False

            if show_child:
                if outcommented_block:
                    contents += outcommented_block._to_prompt_string(
                        show_line_numbers=show_line_numbers,
                    )
                    outcommented_block = None

                contents += child._to_prompt_string(
                    show_span_id=show_new_span_id,
                    show_line_numbers=show_line_numbers,
                    span_marker=SpanMarker.TAG,
                )

                contents += self._to_prompt(
                    code_block=child,
                    exclude_comments=exclude_comments,
                    show_outcommented_code=show_outcommented_code,
                    outcomment_code_comment=outcomment_code_comment,
                    show_span_id=show_span_id,
                    current_span=current_span,
                    show_line_numbers=show_line_numbers,
                    show_all_spans=show_all_spans,
                    only_signatures=only_signatures,
                )
            elif (
                show_outcommented_code
                and not outcommented_block
                and child.type
                not in [
                    CodeBlockType.COMMENT,
                    CodeBlockType.COMMENTED_OUT_CODE,
                    CodeBlockType.SPACE,
                ]
            ):
                outcommented_block = child.create_commented_out_block(
                    outcomment_code_comment
                )
                outcommented_block.start_line = child.start_line

        if show_outcommented_code and outcommented_block:
            contents += outcommented_block._to_prompt_string(
                show_line_numbers=show_line_numbers,
            )

        return contents

    def context_size(self):
        if self.module:
            if self.span_ids is None:
                return self.module.sum_tokens()
            else:
                tokens = 0
                for span_id in self.span_ids:
                    span = self.module.find_span_by_id(span_id)
                    if span:
                        tokens += span.tokens
                return tokens
        else:
            return 0  # TODO: Support context size...

    def has_span(self, span_id: str):
        return span_id in self.span_ids

    def add_spans(
        self,
        span_ids: Set[str],
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ):
        for span_id in span_ids:
            self.add_span(span_id, tokens=tokens, pinned=pinned, add_extra=add_extra)

    def add_span(
        self,
        span_id: str,
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ) -> bool:
        existing_span = next(
            (span for span in self.spans if span.span_id == span_id), None
        )

        if existing_span:
            existing_span.tokens = tokens
            existing_span.pinned = pinned
            return False
        else:
            span = self.module.find_span_by_id(span_id)
            if span:
                self.spans.append(
                    ContextSpan(span_id=span_id, tokens=tokens, pinned=pinned)
                )
                if add_extra:
                    self._add_class_span(span)
                return True
            else:
                logger.warning(
                    f"Tried to add not existing span id {span_id} in file {self.file_path}"
                )
                return False

    def _add_class_span(self, span: BlockSpan):
        if span.initiating_block.type != CodeBlockType.CLASS:
            class_block = span.initiating_block.find_type_in_parents(
                CodeBlockType.CLASS
            )
        elif span.initiating_block.type == CodeBlockType.CLASS:
            class_block = span.initiating_block
        else:
            return

        if not class_block or self.has_span(class_block.belongs_to_span.span_id):
            return

        # Always add init spans like constructors to context
        for child in class_block.children:
            if (
                child.belongs_to_span.span_type == SpanType.INITATION
                and child.belongs_to_span.span_id
                and not self.has_span(child.belongs_to_span.span_id)
            ):
                if child.belongs_to_span.span_id not in self.span_ids:
                    self.spans.append(
                        ContextSpan(span_id=child.belongs_to_span.span_id)
                    )

        if class_block.belongs_to_span.span_id not in self.span_ids:
            self.spans.append(ContextSpan(span_id=class_block.belongs_to_span.span_id))

    def add_line_span(
        self, start_line: int, end_line: int | None = None, add_extra: bool = True
    ) -> list[str]:
        if not self.module:
            logger.warning(f"Could not find module for file {self.file_path}")
            return []

        logger.debug(f"Adding line span {start_line} - {end_line} to {self.file_path}")
        blocks = self.module.find_blocks_by_line_numbers(
            start_line, end_line, include_parents=True
        )

        added_spans = []
        tokens = 0
        for block in blocks:
            if (
                block.belongs_to_span
                and block.belongs_to_span.span_id not in self.span_ids
            ):
                added_spans.append(block.belongs_to_span.span_id)
                self.add_span(block.belongs_to_span.span_id, add_extra=add_extra)
                tokens += block.belongs_to_span.tokens

        logger.info(
            f"Added {added_spans} with {tokens} tokens on lines {start_line} - {end_line} to {self.file_path} to context"
        )
        return added_spans

    def lines_is_in_context(self, start_line: int, end_line: int) -> bool:
        """
        Check if the given line range's start and end points are covered by spans in the context.
        A single span can cover both points, or different spans can cover each point.

        Args:
            start_line (int): Start line number
            end_line (int): End line number

        Returns:
            bool: True if both start and end lines are covered by spans in context, False otherwise
        """
        if self.show_all_spans:
            return True

        if not self.module:
            return False

        start_covered = False
        end_covered = False

        for span in self.spans:
            block_span = self.module.find_span_by_id(span.span_id)
            if block_span:
                if block_span.start_line <= start_line <= block_span.end_line:
                    start_covered = True
                if block_span.start_line <= end_line <= block_span.end_line:
                    end_covered = True
                if start_covered and end_covered:
                    return True

        return False

    def remove_span(self, span_id: str):
        self.spans = [span for span in self.spans if span.span_id != span_id]

    def remove_all_spans(self):
        self.spans = [span for span in self.spans if span.pinned]

    def get_spans(self) -> List[BlockSpan]:
        block_spans = []
        for span in self.spans:
            if not self.module:
                continue

            block_span = self.module.find_span_by_id(span.span_id)
            if block_span:
                block_spans.append(block_span)
        return block_spans

    def get_block_span(self, span_id: str) -> Optional[BlockSpan]:
        if not self.module:
            return None
        for span in self.spans:
            if span.span_id == span_id:
                block_span = self.module.find_span_by_id(span_id)
                if block_span:
                    return block_span
                else:
                    logger.warning(
                        f"Could not find span with id {span_id} in file {self.file_path}"
                    )
        return None

    def get_span(self, span_id: str) -> Optional[ContextSpan]:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_patches(self) -> List[str]:
        """
        Get all patches associated with this ContextFile.
        """
        return self.patches

    def apply(self, other: "ContextFile"):
        """
        Apply another ContextFile onto this one.
        Combines spans and patches, and applies patches to the file content.
        """
        # Combine spans
        existing_span_ids = {span.span_id for span in self.spans}
        for span in other.spans:
            if span.span_id not in existing_span_ids:
                self.spans.append(span)

        # Combine patches
        self._patches.extend(other.get_patches())

        # Apply patches to the file content
        self.apply_patches()

    def apply_patches(self):
        """
        Apply all patches to the file content.
        """
        for patch in self._patches:
            self._apply_patch(patch)


class FileContext(BaseModel):
    _repo: Repository | None = PrivateAttr(None)
    _code_index: CodeIndex = PrivateAttr()
    _runtime: RuntimeEnvironment = PrivateAttr()

    _files: Dict[str, ContextFile] = PrivateAttr(default_factory=dict)
    _test_files: set[str] = PrivateAttr(default_factory=set)
    _max_tokens: int = PrivateAttr(default=8000)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        repo: Repository | None,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        **data,
    ):
        super().__init__(**data)

        self._repo = repo
        self._runtime = runtime
        self._code_index = code_index

        if "_files" not in self.__dict__:
            self.__dict__["_files"] = {}

        if "_max_tokens" not in self.__dict__:
            self.__dict__["_max_tokens"] = data.get("max_tokens", 8000)

    @classmethod
    def from_dir(cls, repo_dir: str, max_tokens: int = 8000):
        from moatless.repository.file import FileRepository

        repo = FileRepository(repo_path=repo_dir)
        instance = cls(max_tokens=max_tokens, repo=repo)
        return instance

    @classmethod
    def from_json(cls, repo_dir: str, json_data: str):
        """
        Create a FileContext instance from JSON data.

        :param repo_dir: The repository directory path.
        :param json_data: A JSON string representing the FileContext data.
        :return: A new FileContext instance.
        """
        data = json.loads(json_data)
        return cls.from_dict(data, repo_dir=repo_dir)

    @classmethod
    def from_dict(
        cls, data: Dict, repo_dir: str | None = None, repo: Repository | None = None
    ):
        if not repo and repo_dir:
            repo = FileRepository(repo_path=repo_dir)
        instance = cls(max_tokens=data.get("max_tokens", 8000), repo=repo)
        instance.load_files_from_dict(data.get("files", []))
        return instance

    def load_files_from_dict(self, files: list[dict]):
        for file_data in files:
            file_path = file_data["file_path"]
            show_all_spans = file_data.get("show_all_spans", False)
            spans = [ContextSpan(**span) for span in file_data.get("spans", [])]

            self._files[file_path] = ContextFile(
                file_path=file_path,
                spans=spans,
                show_all_spans=show_all_spans,
                patch=file_data.get("patch"),
                repo=self._repo,
            )

    def model_dump(self, **kwargs):
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True

        files = [file.model_dump(**kwargs) for file in self.__dict__["_files"].values()]
        return {"max_tokens": self.__dict__["_max_tokens"], "files": files}

    def snapshot(self):
        dict = self.model_dump()
        del dict["max_tokens"]
        return dict

    def restore_from_snapshot(self, snapshot: dict):
        self._files.clear()
        self.load_files_from_dict(snapshot.get("files", []))

    def to_files_with_spans(self) -> List[FileWithSpans]:
        return [
            FileWithSpans(file_path=file_path, span_ids=list(file.span_ids))
            for file_path, file in self._files.items()
        ]

    def add_files_with_spans(self, files_with_spans: List[FileWithSpans]):
        for file_with_spans in files_with_spans:
            self.add_spans_to_context(
                file_with_spans.file_path, set(file_with_spans.span_ids)
            )

    def add_file(
        self, file_path: str, show_all_spans: bool = False, add_extra: bool = True
    ) -> ContextFile:
        if file_path not in self._files:
            self._files[file_path] = ContextFile(
                file_path=file_path,
                spans=[],
                show_all_spans=show_all_spans,
                repo=self._repo,
            )
            if add_extra:
                self._files[file_path]._add_import_span()

        return self._files[file_path]

    def add_file_with_lines(
        self, file_path: str, start_line: int, end_line: Optional[int] = None
    ):
        end_line = end_line or start_line
        if file_path not in self._files:
            self.add_file(file_path)

        self._files[file_path].add_line_span(start_line, end_line)

    def remove_file(self, file_path: str):
        if file_path in self._files:
            del self._files[file_path]

    def exists(self, file_path: str):
        return file_path in self._files

    @property
    def files(self):
        return list(self._files.values())

    def add_spans_to_context(
        self,
        file_path: str,
        span_ids: Set[str],
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ):
        if not self.has_file(file_path):
            context_file = self.add_file(file_path)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            context_file.add_spans(span_ids, tokens, pinned=pinned, add_extra=add_extra)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")

    def add_span_to_context(
        self,
        file_path: str,
        span_id: str,
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ) -> bool:
        if not self.has_file(file_path):
            context_file = self.add_file(file_path)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_span(
                span_id, tokens=tokens, pinned=pinned, add_extra=add_extra
            )
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return False

    def add_line_span_to_context(
        self,
        file_path: str,
        start_line: int,
        end_line: int | None = None,
        add_extra: bool = True,
    ) -> List[str]:
        if not self.has_file(file_path):
            context_file = self.add_file(file_path)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_line_span(start_line, end_line, add_extra=add_extra)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return []

    def remove_span_from_context(
        self, file_path: str, span_id: str, remove_file: bool = False
    ):
        context_file = self.get_context_file(file_path)
        if context_file:
            context_file.remove_span(span_id)

            if not context_file.spans and remove_file:
                self.remove_file(file_path)

    def remove_spans_from_context(
        self, file_path: str, span_ids: List[str], remove_file: bool = False
    ):
        for span_id in span_ids:
            self.remove_span_from_context(file_path, span_id, remove_file)

    def get_spans(self, file_path: str) -> List[BlockSpan]:
        context_file = self.get_context_file(file_path)
        if context_file:
            return context_file.get_spans()
        return []

    def get_span(self, file_path: str, span_id: str) -> Optional[BlockSpan]:
        context_file = self.get_context_file(file_path)
        if context_file:
            return context_file.get_block_span(span_id)
        return None

    def has_span(self, file_path: str, span_id: str):
        context_file = self.get_context_file(file_path)
        if context_file:
            return span_id in context_file.span_ids
        return False

    def apply(self, file_context: "FileContext"):
        """
        Apply a list of FileContext instances, collecting their ContextFiles.
        """
        for context_file in file_context.files:
            file_path = context_file.file_path
            if file_path not in self.files:
                self._files[file_path] = ContextFile(
                    file_path=file_path,
                    repo=self.repo,
                    initial_patch=file_context.generate_full_patch(),
                )

            self._files[file_path].spans.extend(context_file.spans)
            self._files[file_path].show_all_spans = context_file.show_all_spans

    def has_file(self, file_path: str):
        return file_path in self._files and self._files[file_path].spans

    def get_file(self, file_path: str) -> Optional[ContextFile]:
        return self.get_context_file(file_path)

    def file_exists(self, file_path: str):
        context_file = self._files.get(file_path)
        return context_file or self._repo.file_exists(file_path)

    def get_context_file(
        self, file_path: str, add_extra: bool = False
    ) -> Optional[ContextFile]:
        file_path = self._repo.get_relative_path(file_path)
        context_file = self._files.get(file_path)

        if not context_file:
            if not self._repo.file_exists(file_path):
                logger.info(f"get_context_file({file_path}) File not found")
                return None

            if self._repo.is_directory(file_path):
                logger.info(f"get_context_file({file_path}) File is a directory")
                return None

            self.add_file(file_path, add_extra=add_extra)
            context_file = self._files[file_path]

        return context_file

    def get_context_files(self) -> List[ContextFile]:
        file_paths = list(self._files.keys())
        for file_path in file_paths:
            yield self.get_context_file(file_path)

    def context_size(self):
        return sum(file.context_size() for file in self._files.values())

    def available_context_size(self):
        return self._max_tokens - self.context_size()

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        self._repo.save_file(file_path, updated_content)

    def reset(self):
        self._files = {}

    def is_empty(self):
        return not self._files

    def strip_line_breaks_only(self, text):
        return text.lstrip("\n\r").rstrip("\n\r")

    def create_prompt(
        self,
        show_span_ids=False,
        show_line_numbers=False,
        exclude_comments=False,
        show_outcommented_code=False,
        outcomment_code_comment: str = "...",
        files: set | None = None,
        only_signatures: bool = False,
    ):
        file_contexts = []
        for context_file in self.get_context_files():
            if not files or context_file.file_path in files:
                content = context_file.to_prompt(
                    show_span_ids,
                    show_line_numbers,
                    exclude_comments,
                    show_outcommented_code,
                    outcomment_code_comment,
                    only_signatures=only_signatures,
                )
                file_contexts.append(content)

        return "\n\n".join(file_contexts)

    def clone(self):
        dump = self.model_dump()
        cloned_context = FileContext(repo=self._repo)
        cloned_context.load_files_from_dict(dump.get("files", []))
        return cloned_context

    def has_patch(self):
        return any(file.patch for file in self._files.values())

    def generate_git_patch(self) -> str:
        """
        Generates a full patch for all files with changes in the FileContext.
        The patch is formatted like a git diff.

        Returns:
            str: A git-diff-like patch string containing all file changes.
        """
        full_patch = []
        for file_path, context_file in self._files.items():
            if context_file.patch:
                full_patch.append(context_file.patch)

        return "\n".join(full_patch)

    def get_updated_files(
        self, old_context: "FileContext", include_patches: bool = False
    ) -> set[str]:
        """
        Compares this FileContext with an older one and returns a set of files that have been updated.
        Updates include content changes, span additions/removals, and file additions.

        Args:
            old_context: The previous FileContext to compare against

        Returns:
            set[str]: Set of file paths that have different content or spans between the two contexts
        """
        updated_files = set()

        # Check files in current context
        for file_path, current_file in self._files.items():
            old_file = old_context._files.get(file_path)

            if old_file is None:
                # New file added
                updated_files.add(file_path)
            else:
                # Check for content changes
                if include_patches and current_file.content != old_file.content:
                    updated_files.add(file_path)
                    continue

                # Check for span changes
                current_spans = current_file.span_ids
                old_spans = old_file.span_ids
                if current_spans != old_spans:
                    updated_files.add(file_path)

        return updated_files

    def get_context_diff(self, old_context: "FileContext") -> "FileContext":
        diff_context = FileContext(repo=self._repo)

        for file_path, current_file in self._files.items():
            old_file = old_context._files.get(file_path)

            if old_file is None:
                logger.info(f"File {file_path} is new")
                diff_context._files[file_path] = current_file
            else:
                current_spans = current_file.span_ids
                old_spans = old_file.span_ids
                new_spans = current_spans - old_spans
                if new_spans:
                    diff_context._files[file_path] = ContextFile(
                        file_path=file_path,
                        repo=self._repo,
                    )
                    diff_context._files[file_path].spans.extend(
                        [
                            span
                            for span in current_file.spans
                            if span.span_id in new_spans
                        ]
                    )

        return diff_context

    def create_summary(self) -> str:
        """
        Creates a summary of the files and spans in the context.

        Returns:
            str: A formatted summary string listing files and their spans
        """
        if self.is_empty():
            return "No files in context"

        summary = ""
        for context_file in self.get_context_files():
            summary += f"\nFile: {context_file.file_path}\n"

            if context_file.show_all_spans:
                summary += "- Showing all code in file\n"
                continue

            if not context_file.spans:
                continue

            for span in context_file.spans:
                if span.start_line and span.end_line:
                    summary += f"- Lines {span.start_line}-{span.end_line}\n"
                else:
                    summary += f"- {span.span_id}\n"

        return summary

    def add_file_context(self, other_context: "FileContext") -> List[str]:
        """
        Adds spans from another FileContext to the current one and returns newly added span IDs.

        Args:
            other_context: The FileContext to merge into this one

        Returns:
            List[str]: List of newly added span IDs
        """
        new_span_ids = []

        for other_file in other_context.files:
            file_path = other_file.file_path

            if not self.has_file(file_path):
                # Create new file if it doesn't exist
                context_file = self.add_file(file_path)
            else:
                context_file = self.get_context_file(file_path)

            # Add spans that don't already exist
            for span in other_file.spans:
                if context_file.add_span(span.span_id):
                    new_span_ids.append(span.span_id)

            # Copy show_all_spans flag if either context has it enabled
            context_file.show_all_spans = (
                context_file.show_all_spans or other_file.show_all_spans
            )

        return new_span_ids

    def span_count(self) -> int:
        """
        Returns the total number of span IDs across all files in the context.

        Returns:
            int: Total number of span IDs
        """
        span_ids = []
        for file in self._files.values():
            span_ids.extend(file.span_ids)
        return len(span_ids)
