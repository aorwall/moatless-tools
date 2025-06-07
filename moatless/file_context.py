import difflib
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Literal, Any
from unittest import TestResult

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from unidiff import PatchSet

from moatless.artifacts.artifact import ArtifactChange
from moatless.codeblocks import CodeBlockType, get_parser_by_path
from moatless.codeblocks.codeblocks import (
    BlockSpan,
    CodeBlock,
    CodeBlockTypeGroup,
    SpanMarker,
    SpanType,
)
from moatless.codeblocks.module import Module
from moatless.repository import FileRepository
from moatless.repository.git import GitRepository
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.schema import FileWithSpans
from moatless.testing.schema import TestFile
from moatless.utils.file import is_test
from moatless.utils.tokenizer import count_tokens
from moatless.workspace import Workspace

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
        patch (Optional[str]): A Git-formatted patch representing the latest changes applied in this ContextFile.
        spans (List[ContextSpan]): A list of spans associated with this file.
        show_all_spans (bool): A flag to indicate whether to display all spans.
    """

    file_path: str = Field(..., description="The relative path to the file within the repository.")
    patch: Optional[str] = Field(
        None,
        description="Git-formatted patch representing the latest changes applied in this ContextFile.",
    )

    spans: list[ContextSpan] = Field(
        default_factory=list,
        description="List of context spans associated with this file.",
    )
    show_all_spans: bool = Field(False, description="Flag to indicate whether to display all context spans.")

    was_edited: bool = Field(default=False, exclude=True)
    was_viewed: bool = Field(default=False, exclude=True)

    _cached_base_content: Optional[str] = PrivateAttr(None)
    _cached_content: Optional[str] = PrivateAttr(None)
    _cached_module: Optional[Module] = PrivateAttr(None)

    _repo: Repository = PrivateAttr()

    _is_new: bool = PrivateAttr(False)

    def __init__(
        self,
        repo: Optional[Repository],
        **data,
    ):
        """
        Initializes the ContextFile instance.

        Args:
            repo (Optional[Repository]): The repository instance, can be None when reconstructing from dict
        """
        super().__init__(**data)
        self._repo = repo
        self._is_new = False if repo is None else not repo.file_exists(self.file_path)
        self._cached_content = data.get("content", None)  # Store initial content if provided

    def _add_import_span(self):
        # TODO: Initiate module or add this lazily?
        if self.module:
            # Always include init spans like 'imports' to context file
            for child in self.module.children:
                if (child.type == CodeBlockType.IMPORT) and child.belongs_to_span.span_id:
                    self.add_span(child.belongs_to_span.span_id, pinned=True)

    def get_base_content(self) -> str:
        """
        Retrieves the base content of the file.

        Returns:
            str: The base content of the file.

        Raises:
            FileNotFoundError: If the file does not exist in the repository.
        """
        if not self._repo:
            logger.warning("No repo set")
            raise RuntimeError("No repository set on file context")

        if self._cached_base_content is not None:
            return self._cached_base_content

        if not self._repo.file_exists(self.file_path):
            original_content = ""
        else:
            original_content = self._repo.get_file_content(self.file_path)

        # Ensure original_content is a string, not None
        if original_content is None:
            original_content = ""

        self._cached_base_content = original_content

        return self._cached_base_content
    
    @property
    def shadow_mode(self) -> bool:
        if not self._repo:
            return False
        return self._repo.shadow_mode

    @property
    def module(self) -> Module | None:
        if not self._repo:
            return None

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
        if self.patch and self.shadow_mode:
            try:
                self._cached_content = self.apply_patch_to_content(base_content, self.patch)
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

        Returns:
            set[str]: Set of new span IDs added to context
        """
        self.was_edited = True

        if not self.shadow_mode:
            logger.info(f"Saving file {self.file_path} to disk")
            self._repo.save_file(self.file_path, updated_content)
            self._cached_content = None

            if isinstance(self._repo, GitRepository):
                self.patch = self._repo.file_diff(self.file_path)

        else:
            base_content = self.get_base_content()
            self.patch = self.generate_patch(base_content, updated_content)

        new_span_ids = set()

        # If no patch was generated (content identical), return empty set
        if not self.patch:
            return new_span_ids

        try:
            # Track modified lines from patch
            patch_set = PatchSet(io.StringIO(self.patch))
            for patched_file in patch_set:
                for hunk in patched_file:
                    # Get the line range for this hunk's changes
                    modified_start = None
                    modified_end = None

                    for line in hunk:
                        if line.is_added or line.is_removed:
                            # Convert to 0-based line numbers
                            current_line = line.target_line_no if line.is_added else line.source_line_no
                            if current_line is not None:
                                if modified_start is None:
                                    modified_start = current_line
                                modified_end = current_line

                    if modified_start is not None:
                        # Add the modified line span to context
                        span_ids = self.add_line_span(modified_start, modified_end)
                        new_span_ids.update(span_ids)
        except Exception as e:
            # If parsing still fails despite our improved patch generation,
            # fall back to a simple line-by-line comparison to identify changes
            logger.warning(f"Failed to parse patch for {self.file_path}: {e}")

            # Directly compare old and new content to find changed lines
            if base_content and updated_content:
                old_lines = base_content.splitlines()
                new_lines = updated_content.splitlines()

                # Find line numbers where content differs
                for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
                    if old_line != new_line:
                        # Add 1 to convert to 1-based line numbers
                        line_num = i + 1
                        span_ids = self.add_line_span(line_num)
                        new_span_ids.update(span_ids)

                # Handle case where file was lengthened
                if len(new_lines) > len(old_lines):
                    for i in range(len(old_lines), len(new_lines)):
                        line_num = i + 1
                        span_ids = self.add_line_span(line_num)
                        new_span_ids.update(span_ids)

        self._cached_content = None
        self._cached_module = None

        return new_span_ids

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
        if not patch.strip():
            return content

        try:
            # Try using the unidiff library first
            patch_set = PatchSet(io.StringIO(patch))
            patched_content = content

            for patched_file in patch_set:
                patched_file_path = patched_file.path
                # Correctly strip 'a/' or 'b/' prefixes
                if patched_file_path.startswith("a/") or patched_file_path.startswith("b/"):
                    patched_file_path = patched_file_path[2:]
                if os.path.normpath(patched_file_path) == os.path.normpath(self.file_path):
                    patched_content = self._apply_patched_file(patched_content, patched_file)
                    break
            else:
                raise Exception(f"Patch does not contain changes for file {self.file_path}")

            return patched_content

        except Exception as e:
            logger.warning(f"Failed to apply patch using unidiff: {e}")
            # Fallback to a simple manual patch application
            return self._apply_patch_manually(content, patch)

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
        line_no = 0

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
                                f"Patch context mismatch: Line {line_no} is beyond end of file ({len(content_lines)} lines)"
                            )
                        elif line.value.strip() and content_lines[line_no].strip() != line.value.strip():
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

    def _apply_patch_manually(self, content: str, patch: str) -> str:
        """
        Apply a patch manually when unidiff fails.
        This is a fallback mechanism for problematic patches.
        """
        # If the patch doesn't look like a valid diff, return original content
        if not patch.startswith("---") or "+++" not in patch or "@@" not in patch:
            logger.warning("Patch doesn't look like a valid diff, returning original content")
            return content

        content_lines = content.splitlines()
        result_lines = content_lines.copy()

        # Find all hunks in the patch
        hunks = []
        current_hunk = None
        hunk_start_line = 0
        target_lines = []

        for line in patch.splitlines():
            if line.startswith("@@"):
                # Parse the hunk header to extract line numbers
                try:
                    # Parse something like "@@ -27,11 +27,11 @@"
                    hunk_header = line.split("@@")[1].strip()
                    old_range = hunk_header.split(" ")[0]
                    hunk_start_line = int(old_range.split(",")[0].lstrip("-"))

                    # Start a new hunk
                    if current_hunk:
                        hunks.append((hunk_start_line, current_hunk))
                    current_hunk = []
                    target_lines = []
                except Exception as e:
                    logger.warning(f"Failed to parse hunk header: {e}")
                    continue
            elif current_hunk is not None:
                if line.startswith("+"):
                    # Added line
                    target_lines.append(line[1:])
                elif line.startswith("-"):
                    # Removed line - keep track for context matching
                    current_hunk.append(line[1:])
                elif not line.startswith("\\"):
                    # Context line - should match in both versions
                    current_hunk.append(line)
                    target_lines.append(line)

        # Add the last hunk
        if current_hunk:
            hunks.append((hunk_start_line, current_hunk))

        # Apply hunks in reverse order to avoid line number shifting
        for start_line, hunk_lines in sorted(hunks, reverse=True):
            # Find the best match for this hunk in the content
            best_match_idx = self._find_best_match_for_hunk(content_lines, hunk_lines, start_line - 1)

            if best_match_idx >= 0:
                # Replace content at the matched position
                hunk_target_idx = hunks.index((start_line, hunk_lines))
                target_content = target_lines[hunk_target_idx]

                # Delete old lines
                del result_lines[best_match_idx : best_match_idx + len(hunk_lines)]

                # Insert new content
                for i, line in enumerate(target_content):
                    result_lines.insert(best_match_idx + i, line)
            else:
                logger.warning(f"Could not find match for hunk at line {start_line}")

        return "\n".join(result_lines)

    def _find_best_match_for_hunk(self, content_lines, hunk_lines, expected_line_idx):
        """
        Find the best position in content_lines that matches the hunk lines.
        Uses both the expected line index and a fuzzy matching approach.
        """
        if not hunk_lines:
            return -1

        # First try exact match at the expected position
        if expected_line_idx < len(content_lines):
            # Check if hunk can be applied at expected position
            if all(
                i + expected_line_idx < len(content_lines)
                and (
                    hl == content_lines[i + expected_line_idx]
                    or hl.strip() == content_lines[i + expected_line_idx].strip()
                )
                for i, hl in enumerate(hunk_lines)
            ):
                return expected_line_idx

        # If exact match failed, try fuzzy matching
        # Look for the first line of the hunk
        if not hunk_lines[0].strip():
            # Skip empty lines for matching
            first_line = next((line for line in hunk_lines if line.strip()), "")
        else:
            first_line = hunk_lines[0]

        if not first_line:
            return -1

        # Try to find this line in the content
        for i, line in enumerate(content_lines):
            if first_line == line or first_line.strip() == line.strip():
                # Check if subsequent lines also match
                matches = True
                for j, hunk_line in enumerate(hunk_lines):
                    if i + j >= len(content_lines):
                        matches = False
                        break
                    content_line = content_lines[i + j]
                    if hunk_line != content_line and hunk_line.strip() != content_line.strip():
                        matches = False
                        break

                if matches:
                    return i

        # If no good match found, return -1
        return -1

    def generate_full_patch(self) -> str:
        """
        Generates a full Git-formatted patch from the original content to the current content.

        Returns:
            str: The generated full patch as a string.
        """
        original_content = self._repo.get_file_content(self.file_path)
        current_content = self.content

        patch = self.generate_patch(original_content, current_content)
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
        # Handle empty content cases
        if old_content is None:
            old_content = ""
        if new_content is None:
            new_content = ""

        # If content is identical, return empty patch
        if old_content == new_content:
            return ""

        # Pre-process both strings to ensure consistent line endings
        # Split by line but don't keep the line endings - we'll normalize them
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        # Generate the unified diff with 3 lines of context (standard for git)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="a/" + self.file_path,
                tofile="b/" + self.file_path,
                lineterm="",
                n=3,
            )
        )

        # Early return if no differences
        if not diff_lines:
            return ""

        # Join with newlines and add a final newline
        patch = "\n".join(diff_lines) + "\n"

        try:
            # Validate by test-parsing the patch
            PatchSet(io.StringIO(patch))

            # If we got here, the patch is valid
            return patch
        except Exception as e:
            logger.debug(f"Failed to generate valid patch with simplified method: {e}")

            # Fallback to the original method with context size variation
            return self._generate_fallback_patch(old_content, new_content)

    def _generate_fallback_patch(self, old_content: str, new_content: str) -> str:
        """Fallback method for patch generation when the simple approach fails."""
        # Pre-process both strings to ensure consistent line endings
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        # Ensure both old and new content end with a newline to prevent issues
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        # Try progressively larger context sizes if needed
        for context_lines in [3, 5, 7]:
            try:
                # Generate the unified diff with specified context lines
                diff_lines = list(
                    difflib.unified_diff(
                        old_lines,
                        new_lines,
                        fromfile="a/" + self.file_path,
                        tofile="b/" + self.file_path,
                        lineterm="",  # We'll add newlines manually
                        n=context_lines,
                    )
                )

                # Early return if no differences
                if not diff_lines:
                    return ""

                # Process the diff to ensure consistent formatting for unidiff
                processed_diff = self._process_diff_for_unidiff(diff_lines)

                # Validate by test-parsing the patch
                PatchSet(io.StringIO(processed_diff))

                # If we got here, the patch is valid
                return processed_diff

            except Exception as e:
                logger.debug(f"Failed to generate valid patch with context={context_lines}: {e}")
                continue

        # If all attempts failed, try with a very large context to capture all content
        try:
            # Use a manual approach as a fallback
            return self._generate_manual_patch(old_content, new_content)
        except Exception as e:
            logger.warning(f"All patch generation methods failed: {e}. Using raw diff.")
            # Last resort: just use the raw diff with newlines and hope for the best
            diff_text = (
                "\n".join(
                    difflib.unified_diff(
                        old_content.splitlines(),
                        new_content.splitlines(),
                        fromfile="a/" + self.file_path,
                        tofile="b/" + self.file_path,
                        lineterm="",
                        n=3,
                    )
                )
                + "\n"
            )
            return diff_text

    def _process_diff_for_unidiff(self, diff_lines):
        """
        Process diff lines to ensure they're properly formatted for unidiff parsing.
        Especially handles cases with consecutive blank lines.
        """
        if not diff_lines:
            return ""

        # Join with newlines and add a final newline
        diff_text = "\n".join(diff_lines) + "\n"

        # Fix hunk headers to match the actual content
        # This is critical for handling consecutive blank lines correctly
        fixed_lines = []
        current_hunk_start = None
        hunk_old_count = 0
        hunk_new_count = 0
        actual_old_count = 0
        actual_new_count = 0

        for i, line in enumerate(diff_text.splitlines()):
            # Detect hunk headers
            if line.startswith("@@"):
                # If we were processing a hunk, fix its header before starting a new one
                if current_hunk_start is not None:
                    # Fix the previous hunk header
                    header_parts = fixed_lines[current_hunk_start].split()
                    header_parts[1] = f"-{hunk_old_count},{actual_old_count}"
                    header_parts[2] = f"+{hunk_new_count},{actual_new_count}"
                    fixed_lines[current_hunk_start] = " ".join(header_parts)

                # Start tracking a new hunk
                current_hunk_start = len(fixed_lines)
                fixed_lines.append(line)  # Add the header as a placeholder to be fixed later

                # Parse the hunk header to get expected counts
                try:
                    old_range = line.split()[1]
                    new_range = line.split()[2]
                    hunk_old_count = int(old_range.split(",")[0].lstrip("-"))
                    hunk_new_count = int(new_range.split(",")[0].lstrip("+"))
                    actual_old_count = 0
                    actual_new_count = 0
                except (IndexError, ValueError):
                    # If parsing fails, use default values
                    hunk_old_count = 0
                    hunk_new_count = 0

            else:
                fixed_lines.append(line)

                # Count lines
                if current_hunk_start is not None:
                    if line.startswith("+"):
                        actual_new_count += 1
                    elif line.startswith("-"):
                        actual_old_count += 1
                    elif not line.startswith("\\"):  # Ignore "No newline" markers
                        # Context lines count for both old and new
                        actual_old_count += 1
                        actual_new_count += 1

        # Fix the last hunk header if there was one
        if current_hunk_start is not None:
            header_parts = fixed_lines[current_hunk_start].split()
            header_parts[1] = f"-{hunk_old_count},{actual_old_count}"
            header_parts[2] = f"+{hunk_new_count},{actual_new_count}"
            fixed_lines[current_hunk_start] = " ".join(header_parts)

        return "\n".join(fixed_lines) + "\n"

    def _generate_manual_patch(self, old_content, new_content):
        """
        Generate a manual patch as a fallback when diff doesn't work.
        This creates a single hunk that replaces the entire file.
        """
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        # Create a header for the entire file
        header = f"--- a/{self.file_path}\n+++ b/{self.file_path}\n@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n"

        # Create the content lines
        content = []
        for line in old_lines:
            content.append(f"-{line}")
        for line in new_lines:
            content.append(f"+{line}")

        return header + "\n".join(content) + "\n"

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
        max_tokens: Optional[int] = None,
    ):
        if self.module:
            if not self.show_all_spans and self.span_ids is not None and len(self.span_ids) == 0:
                logger.warning(f"No span ids provided for {self.file_path}, return empty")
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
                max_tokens=max_tokens,
            )
        else:
            code = self._to_prompt_with_line_spans(show_span_id=show_span_ids)

        result = f"{self.file_path}\n```\n{code}\n```\n"

        # Check if result exceeds max_tokens
        if max_tokens and count_tokens(result) > max_tokens:
            logger.warning(f"Content for {self.file_path} exceeded max_tokens ({max_tokens})")
            return ""

        return result

    def _find_span(self, codeblock: CodeBlock) -> Optional[ContextSpan]:
        if not codeblock.belongs_to_span:
            return None

        for span in self.spans:
            if codeblock.belongs_to_span.span_id == span.span_id:
                return span

        return None

    def _within_span(self, line_no: int) -> Optional[ContextSpan]:
        for span in self.spans:
            if span.start_line and span.end_line and span.start_line <= line_no <= span.end_line:
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
        max_tokens: Optional[int] = None,
        current_tokens: int = 0,
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

            # Check if adding this block would exceed max_tokens
            if max_tokens:
                if current_tokens + child.tokens > max_tokens:
                    logger.debug("Stopping at child block as it would exceed max_tokens")
                    break

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
                    if child.type.group == CodeBlockTypeGroup.IMPLEMENTATION and child.type not in [
                        CodeBlockType.COMPOUND,
                        CodeBlockType.DEPENDENT_CLAUSE,
                    ]:
                        child_tokens = child.sum_tokens()
                    else:
                        child_tokens = child.tokens

                    if current_span.tokens + child_tokens <= child_span.tokens:
                        show_child = True

                    current_span.tokens += child_tokens

            elif (not child.belongs_to_span or child.belongs_to_any_span not in self.spans) and child.has_any_span(
                self.span_ids
            ):
                show_child = True

                if child.belongs_to_span and current_span.span_id != child.belongs_to_span.span_id:
                    show_new_span_id = show_span_id
                    current_span = CurrentPromptSpan(child.belongs_to_span.span_id)

            if self.show_all_spans or show_all_spans:
                show_child = True

            if only_signatures and child.type.group != CodeBlockTypeGroup.STRUCTURE:
                show_child = False

            if show_child:
                if outcommented_block:
                    block_content = outcommented_block._to_prompt_string(
                        show_line_numbers=show_line_numbers,
                    )
                    contents += block_content
                    current_tokens += count_tokens(block_content)
                    outcommented_block = None

                block_content = child._to_prompt_string(
                    show_span_id=show_new_span_id,
                    show_line_numbers=show_line_numbers,
                    span_marker=SpanMarker.TAG,
                )
                contents += block_content
                current_tokens += count_tokens(block_content)

                child_content = self._to_prompt(
                    code_block=child,
                    exclude_comments=exclude_comments,
                    show_outcommented_code=show_outcommented_code,
                    outcomment_code_comment=outcomment_code_comment,
                    show_span_id=show_span_id,
                    current_span=current_span,
                    show_line_numbers=show_line_numbers,
                    show_all_spans=show_all_spans,
                    only_signatures=only_signatures,
                    max_tokens=max_tokens,
                    current_tokens=current_tokens,
                )
                contents += child_content
                current_tokens += count_tokens(child_content)

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
                outcommented_block = child.create_commented_out_block(outcomment_code_comment)
                outcommented_block.start_line = child.start_line

        if show_outcommented_code and outcommented_block:
            block_content = outcommented_block._to_prompt_string(
                show_line_numbers=show_line_numbers,
            )
            contents += block_content
            current_tokens += count_tokens(block_content)

        return contents

    def set_patch(self, patch: str):
        self.patch = patch
        self._cached_content = None
        self._cached_module = None
        self.was_edited = True

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

    def count_line_changes(self, old_content: Optional[str] = None) -> dict[str, Any]:
        """
        Count the number of added and removed lines between the current file content and old content.

        Args:
            old_content: The previous content of the file. If None, will compare with base content.

        Returns:
            dict: A dictionary with keys 'added_lines', 'removed_lines', and 'token_count'
        """
        if old_content is None:
            old_content = self.get_base_content()

        current_content = self.content

        patch = self.generate_patch(old_content, current_content)
        if patch:
            patch_lines = patch.split("\n")
            additions = sum(1 for line in patch_lines if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in patch_lines if line.startswith("-") and not line.startswith("---"))
            return {"added_lines": additions, "removed_lines": deletions, "diff": patch}
        else:
            return {"added_lines": 0, "removed_lines": 0}

    def has_span(self, span_id: str):
        return span_id in self.span_ids

    def add_spans(
        self,
        span_ids: set[str],
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ):
        for span_id in span_ids:
            self.add_span(span_id, tokens=tokens, pinned=pinned, add_extra=add_extra)

    def add_span(
        self,
        span_id: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        tokens: Optional[int] = None,
        pinned: bool = False,
        add_extra: bool = True,
    ) -> bool:
        self.was_viewed = True
        existing_span = next((span for span in self.spans if span.span_id == span_id), None)

        if existing_span:
            existing_span.tokens = tokens
            existing_span.pinned = pinned
            return False
        else:
            span = self.module.find_span_by_id(span_id)
            if span:
                self.spans.append(
                    ContextSpan(
                        span_id=span_id,
                        start_line=start_line,
                        end_line=start_line,
                        tokens=tokens,
                        pinned=pinned,
                    )
                )
                if add_extra:
                    self._add_class_span(span)
                return True
            else:
                logger.warning(f"Tried to add not existing span id {span_id} in file {self.file_path}")
                return False

    def _add_class_span(self, span: BlockSpan):
        if span.initiating_block.type != CodeBlockType.CLASS:
            class_block = span.initiating_block.find_type_in_parents(CodeBlockType.CLASS)
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
                    self.spans.append(ContextSpan(span_id=child.belongs_to_span.span_id))

        if class_block.belongs_to_span.span_id not in self.span_ids:
            self.spans.append(ContextSpan(span_id=class_block.belongs_to_span.span_id))

    def add_line_span(self, start_line: int, end_line: int | None = None, add_extra: bool = True) -> list[str]:
        self.was_viewed = True

        if not self.module:
            logger.warning(f"Could not find module for file {self.file_path}")
            return []

        logger.debug(f"Adding line span {start_line} - {end_line} to {self.file_path}")
        blocks = self.module.find_blocks_by_line_numbers(start_line, end_line, include_parents=True)

        added_spans = []
        for block in blocks:
            if block.belongs_to_span and block.belongs_to_span.span_id not in self.span_ids:
                added_spans.append(block.belongs_to_span.span_id)
                self.add_span(
                    block.belongs_to_span.span_id,
                    start_line=start_line,
                    end_line=end_line,
                    add_extra=add_extra,
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

    def get_spans(self) -> list[BlockSpan]:
        block_spans = []
        for span in self.spans:
            if not self.module:
                continue

            block_span = self.module.find_span_by_id(span.span_id)
            if block_span:
                block_spans.append(block_span)
        return block_spans

    def get_span(self, span_id: str) -> Optional[ContextSpan]:
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    @property
    def is_new(self) -> bool:
        """
        Returns whether this file is newly created in the context.

        Returns:
            bool: True if the file is new, False otherwise
        """
        return self._is_new


class FileContext(BaseModel):
    show_code_blocks: bool = Field(
        False,
        description="Whether to show the parsed code blocks in the response or just the line span.",
    )
    _repo: Repository | None = PrivateAttr(None)
    _runtime: RuntimeEnvironment | None = PrivateAttr(None)

    _files: dict[str, ContextFile] = PrivateAttr(default_factory=dict)
    _test_files: dict[str, TestFile] = PrivateAttr(default_factory=dict)  # Changed to Dict
    _max_tokens: int = PrivateAttr(default=8000)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
        **data,
    ):
        super().__init__(**data)

        self._repo = repo
        self._runtime = runtime

        if "_files" not in self.__dict__:
            self.__dict__["_files"] = {}

        if "_test_files" not in self.__dict__:
            self.__dict__["_test_files"] = {}

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
        cls,
        data: dict,
        repo_dir: str | None = None,
        repo: Repository | None = None,
        runtime: RuntimeEnvironment | None = None,
    ):
        if not repo and repo_dir:
            repo = FileRepository(repo_path=repo_dir)
        instance = cls(
            max_tokens=data.get("max_tokens", 8000),
            repo=repo,
            runtime=runtime
        )
        instance.load_files_from_dict(data.get("files", []), test_files=data.get("test_files", []))
        return instance

    @property
    def shadow_mode(self) -> bool:
        if not self._repo:
            return False
        return self._repo.shadow_mode

    def load_files_from_dict(self, files: list[dict], test_files: list[dict] | None = None):
        """
        Loads files and test files from a dictionary representation.

        Args:
            files (list[dict]): List of file data dictionaries
            test_files (list[dict] | None): List of test file data dictionaries
        """
        # Load regular files
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
                shadow_mode=self.shadow_mode,
            )

        # Load test files
        if test_files:
            for test_file_data in test_files:
                file_path = test_file_data["file_path"]
                self._test_files[file_path] = TestFile(**test_file_data)

    def model_dump(self, **kwargs):
        """
        Dumps the model to a dictionary, including files and test files.
        """
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True

        files = [file.model_dump(**kwargs) for file in self._files.values()]
        test_files = [test_file.model_dump(**kwargs) for test_file in self._test_files.values()]

        return {
            "max_tokens": self.__dict__["_max_tokens"],
            "files": files,
            "test_files": test_files,
            "shadow_mode": self.shadow_mode,
        }

    @property
    def repository(self) -> Repository:
        if not self._repo:
            raise ValueError("Repository is not set")
        return self._repo

    @repository.setter
    def repository(self, repo: Repository):
        self._repo = repo
        for file in self._files.values():
            file._repo = repo

    @property
    def workspace(self):
        raise AttributeError("workspace property is write-only")

    @workspace.setter
    def workspace(self, workspace: Workspace):
        self.repository = workspace.repository
        self._runtime = workspace.runtime

    def to_files_with_spans(self) -> list[FileWithSpans]:
        return [
            FileWithSpans(file_path=file_path, span_ids=list(file.span_ids)) for file_path, file in self._files.items()
        ]

    def add_files_with_spans(self, files_with_spans: list[FileWithSpans]):
        for file_with_spans in files_with_spans:
            self.add_spans_to_context(file_with_spans.file_path, set(file_with_spans.span_ids))

    def add_file(self, file_path: str, show_all_spans: bool = False, add_extra: bool = True) -> ContextFile:
        if file_path not in self._files:
            self._files[file_path] = ContextFile(
                file_path=file_path,
                spans=[],
                show_all_spans=show_all_spans,
                repo=self._repo,
                shadow_mode=self.shadow_mode,
            )
            if add_extra:
                self._files[file_path]._add_import_span()

        return self._files[file_path]

    def add_file_with_lines(self, file_path: str, start_line: int, end_line: Optional[int] = None):
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
    def has_runtime(self):
        return bool(self._runtime)

    @property
    def files(self):
        return list(self._files.values())

    @property
    def file_paths(self):
        return list(self._files.keys())

    @property
    def test_files(self):
        return list(self._test_files.values())

    def add_spans_to_context(
        self,
        file_path: str,
        span_ids: set[str],
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
            context_file = self.add_file(file_path, add_extra=add_extra)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_span(span_id, tokens=tokens, pinned=pinned, add_extra=add_extra)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return False

    def add_line_span_to_context(
        self,
        file_path: str,
        start_line: int,
        end_line: int | None = None,
        add_extra: bool = True,
    ) -> list[str]:
        if not self.has_file(file_path):
            context_file = self.add_file(file_path, add_extra=add_extra)
        else:
            context_file = self.get_context_file(file_path)

        if context_file:
            return context_file.add_line_span(start_line, end_line, add_extra=add_extra)
        else:
            logger.warning(f"Could not find file {file_path} in the repository")
            return []

    def has_file(self, file_path: str):
        return file_path in self._files and (
            self._files[file_path].spans or self._files[file_path].show_all_spans or not self._files[file_path].content
        )

    def get_file(self, file_path: str) -> Optional[ContextFile]:
        return self.get_context_file(file_path)

    def file_exists(self, file_path: str):
        context_file = self._files.get(file_path)
        return context_file or self.repository.file_exists(file_path)

    def is_directory(self, file_path: str):
        return self.repository.is_directory(file_path)

    def get_context_file(self, file_path: str, add_extra: bool = False) -> Optional[ContextFile]:
        if self._repo and hasattr(self._repo, "get_relative_path"):
            file_path = self.repository.get_relative_path(file_path)

        context_file = self._files.get(file_path)

        if not context_file:
            if not self.repository.file_exists(file_path):
                logger.info(f"get_context_file({file_path}) File not found")
                return None

            if self.repository.is_directory(file_path):
                logger.info(f"get_context_file({file_path}) File is a directory")
                return None

            self.add_file(file_path, add_extra=add_extra)
            context_file = self._files[file_path]

        return context_file

    def get_context_files(self) -> list[ContextFile]:
        """
        Returns all context files that exist in the repository.

        Returns:
            list[ContextFile]: A list of all context files
        """
        file_paths = list(self._files.keys())
        result = []
        for file_path in file_paths:
            context_file = self.get_context_file(file_path)
            if context_file is not None:
                result.append(context_file)
        return result

    def context_size(self):
        if self._repo:
            content = self.create_prompt(
                show_span_ids=False,
                show_line_numbers=True,
                show_outcommented_code=True,
                outcomment_code_comment="...",
                only_signatures=False,
            )
            return count_tokens(content)

        # TODO: This doesnt give accure results. Will count tokens in the generated prompt instead
        # sum(file.context_size() for file in self._files.values())
        return 0

    def available_context_size(self):
        return self._max_tokens - self.context_size()

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        if updated_content:
            self.repository.save_file(file_path, updated_content)

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
        max_tokens: Optional[int] = None,
    ):
        file_contexts = []
        current_tokens = 0

        for context_file in self.get_context_files():
            if not files or context_file.file_path in files:
                content = context_file.to_prompt(
                    show_span_ids,
                    show_line_numbers,
                    exclude_comments,
                    show_outcommented_code,
                    outcomment_code_comment,
                    only_signatures=only_signatures,
                    max_tokens=max_tokens,
                )

                if max_tokens:
                    content_tokens = count_tokens(content)
                    if current_tokens + content_tokens > max_tokens:
                        logger.warning(f"Skipping {context_file.file_path} as it would exceed max_tokens")
                        break
                    current_tokens += content_tokens

                if content:  # Only add non-empty content
                    file_contexts.append(content)

        return "\n\n".join(file_contexts)

    def clone(self):
        dump = self.model_dump(exclude={"files": {"__all__": {"was_edited", "was_viewed"}}})
        cloned_context = FileContext(repo=self._repo, runtime=self._runtime, shadow_mode=self.shadow_mode)
        cloned_context.load_files_from_dict(files=dump.get("files", []), test_files=dump.get("test_files", []))
        return cloned_context

    def has_patch(self, ignore_tests: bool = False):
        """
        Checks if any files in the context have patches.

        Args:
            ignore_tests: Whether to ignore test files when checking for patches

        Returns:
            bool: True if any file has a patch, False otherwise
        """
        return any(file.patch for file in self._files.values() if not ignore_tests or not is_test(file.file_path))

    def has_test_patch(self):
        return any(file.patch for file in self._files.values() if is_test(file.file_path))

    def generate_git_patch(self, ignore_tests: bool = False) -> str:
        """
        Generates a full patch for all files with changes in the FileContext.
        The patch is formatted like a git diff.

        Args:
            ignore_tests: Whether to ignore test files when generating the patch

        Returns:
            str: A git-diff-like patch string containing all file changes.
        """
        full_patch = []
        for file_path, context_file in self._files.items():
            if ignore_tests and is_test(file_path):
                continue
            if context_file.patch:
                full_patch.append(context_file.patch)

        return "\n".join(full_patch)

    def get_updated_files(self, old_context: "FileContext", include_patches: bool = True) -> set[str]:
        """
        Compares this FileContext with an older one and returns a set of files that have been updated.
        Updates include content changes, span additions/removals, and file additions.

        Args:
            old_context: The previous FileContext to compare against
            include_patches: Whether to include files with different content (patches)

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

    def get_artifact_changes(
        self, old_context: "FileContext", actor: Literal["user", "assistant"] = "assistant"
    ) -> list[ArtifactChange]:
        """
        Compares this FileContext with an older one and returns a list of ArtifactChange objects for files
        that have been added, updated, or removed.

        Args:
            old_context: The previous FileContext to compare against
            actor: Who made the changes, either "user" or "assistant"

        Returns:
            list[ArtifactChange]: List of ArtifactChange objects representing the changes between contexts
        """
        artifact_changes = []

        # Check files in current context for added or updated files
        for file_path, current_file in self._files.items():
            old_file = old_context._files.get(file_path)

            if old_file is None:
                properties = {
                    "token_count": current_file.context_size(),
                }

                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=file_path,
                        artifact_type="file",
                        change_type="added",
                        properties=properties,
                        actor=actor,
                    )
                )
            elif current_file.patch != old_file.patch:
                properties = current_file.count_line_changes(old_file.content)
                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=file_path,
                        artifact_type="file",
                        change_type="updated",
                        properties=properties,
                        actor=actor,
                    )
                )
            elif current_file.span_ids != old_file.span_ids:
                properties = {
                    "token_count": current_file.context_size() - old_file.context_size(),
                }

                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=file_path,
                        artifact_type="file",
                        change_type="added",
                        properties=properties,
                        actor=actor,
                    )
                )

        for file_path, old_file in old_context._files.items():
            if file_path not in self._files:
                change_stats = {
                    "token_count": -old_file.context_size(),
                }

                artifact_changes.append(
                    ArtifactChange(
                        artifact_id=file_path,
                        artifact_type="file",
                        change_type="removed",
                        properties=change_stats,
                        actor=actor,
                    )
                )

        return artifact_changes

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
                        shadow_mode=self.shadow_mode,
                    )
                    diff_context._files[file_path].spans.extend(
                        [span for span in current_file.spans if span.span_id in new_spans]
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

        summary = []
        for context_file in self.get_context_files():
            # Get file stats
            tokens = context_file.context_size()

            # Get patch stats if available
            patch_stats = ""
            if context_file.patch:
                patch_lines = context_file.patch.split("\n")
                additions = sum(1 for line in patch_lines if line.startswith("+") and not line.startswith("+++"))
                deletions = sum(1 for line in patch_lines if line.startswith("-") and not line.startswith("---"))
                patch_stats = f" (+{additions}/-{deletions})"

            summary.append(f"\n### {context_file.file_path}")
            summary.append(f"- Tokens: {tokens}{patch_stats}")

            if context_file.show_all_spans:
                summary.append("- Showing all code in file")
                continue

            if context_file.spans:
                spans = []
                for span in context_file.spans:
                    if span.start_line and span.end_line:
                        spans.append(f"{span.start_line}-{span.end_line}")
                    else:
                        spans.append(span.span_id)
                summary.append(f"- Spans: {', '.join(spans)}")

        return "\n".join(summary)

    def add_file_context(self, other_context: "FileContext") -> bool:
        """
        Adds spans from another FileContext to the current one and returns newly added span IDs.
        Also copies over cached content and module if they exist.

        Args:
            other_context: The FileContext to merge into this one

        Returns:
            List[str]: List of newly added span IDs
        """
        added_new_spans = False

        for other_file in other_context.files:
            file_path = other_file.file_path

            if not self.has_file(file_path):
                # Create new file if it doesn't exist
                context_file = self.add_file(file_path)
            else:
                context_file = self.get_context_file(file_path)

            if context_file:
                # Copy show_all_spans flag if either context has it enabled
                if not context_file.show_all_spans and other_file.show_all_spans:
                    added_new_spans = True
                    context_file.show_all_spans = other_file.show_all_spans
                else:
                    for span in other_file.spans:
                        if context_file.add_span(span.span_id):
                            added_new_spans = True

        return added_new_spans

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

    def add_test_file(self, file_path: str, test_results: list[TestResult]) -> TestFile:
        if file_path in self._test_files.keys():
            self._test_files[file_path].test_results = test_results
        else:
            self._test_files[file_path] = TestFile(file_path=file_path, test_results=test_results)

        return self._test_files[file_path]

    def add_test_files(self, test_files: list[TestFile]):
        for test_file in test_files:
            self.add_test_file(test_file.file_path, test_file.test_results)

    def get_test_summary(self) -> str:
        return TestFile.get_test_summary(self._test_files.values())

    def get_test_counts(self) -> tuple[int, int, int]:
        return TestFile.get_test_counts(self._test_files.values())

    def get_test_failure_details(self, max_tokens: int = 8000, max_chars_per_test: int = 2000) -> str:
        return TestFile.get_test_failure_details(self._test_files.values(), max_tokens, max_chars_per_test)

    def get_test_status(self):
        return TestFile.get_test_status(self._test_files.values())

    def was_edited(self) -> bool:
        """
        Checks if any files in the context were edited.

        Returns:
            bool: True if any file has been edited, False otherwise
        """
        return any(file.was_edited for file in self._files.values())

    def get_edited_files(self) -> list[str]:
        """
        Returns a list of file paths that have been edited in the context.
        A file is considered edited if it has changes (patch) but is not new.

        Returns:
            List[str]: List of edited file paths
        """
        return [file_path for file_path, file in self._files.items() if file.was_edited and not file.is_new]

    def get_created_files(self) -> list[str]:
        """
        Returns a list of file paths that have been newly created in the context.

        Returns:
            List[str]: List of created file paths
        """
        return [file_path for file_path, file in self._files.items() if file.is_new]

    def persist(self):
        """Persist all files in the context by saving them to the repository"""
        for file_path, file in self._files.items():
            if file.patch:
                self.repository.save_file(file_path, file.content)
