import os
from typing import List, Dict, Optional

from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import (
    _print_content,
    SpanMarker,
    _is_span_within_block,
)
from moatless.types import ContextFile, Span

_parser = PythonParser()


class ContextFile:

    def __init__(self, file_path: str, content: str, spans: Dict[str, Span] = None):
        self._file_path = file_path

        self._module_block = _parser.parse(content)
        self._show_header_types = [CodeBlockType.IMPORT]

        self._spans = spans

        self._span_index = {}
        self._span_marker = SpanMarker.COMMENT

    def to_prompt(self):
        return self._to_prompt(self._module_block)

    def _to_prompt(self, codeblock: CodeBlock) -> str:
        span_id = None

        _current_span = self._span_by_start_line(codeblock.start_line)
        if _current_span:
            span_id = _current_span.span_id
        elif codeblock.is_indexed:
            span_id = codeblock.path_string()

        contents = _print_content(codeblock, span_id=span_id)

        has_outcommented_code = False
        for child in codeblock.children:
            show_child = _is_span_within_block(
                child, spans
            ) or self._is_block_within_spans(
                child, spans
            )  # TODO Optimize
            if spans[0].end_line < child.start_line:
                spans = spans[1:]

                if not spans:
                    break

            if show_child and spans:
                if (
                    outcomment_code_comment
                    and has_outcommented_code
                    and child.type
                    not in [
                        CodeBlockType.COMMENT,
                        CodeBlockType.COMMENTED_OUT_CODE,
                    ]
                ):
                    contents += child.create_commented_out_block(
                        outcomment_code_comment
                    ).to_string()

                contents += self._to_prompt(child, spans)
                has_outcommented_code = False
            else:
                has_outcommented_code = True

        if (
            outcomment_code_comment
            and has_outcommented_code
            and child.type
            not in [
                CodeBlockType.COMMENT,
                CodeBlockType.COMMENTED_OUT_CODE,
            ]
        ):
            contents += child.create_commented_out_block(
                outcomment_code_comment
            ).to_string()

        return contents

    def _span_by_start_line(self, start_line: int) -> Optional[Span]:
        for span in self._spans:
            if span.start_line == start_line:
                return span

        return None

    def _is_span_within_block(self, codeblock: CodeBlock, spans: List[Span]) -> bool:
        for span in spans:
            if (
                span.start_line
                and span.start_line >= codeblock.start_line
                and span.end_line <= codeblock.end_line
            ):
                return True

        return False

    def _is_block_within_spans(self, codeblock: CodeBlock, spans: List[Span]) -> bool:
        for span in spans:
            if (
                codeblock.full_path()[: len(codeblock.full_path())]
                == spans[0].block_path
            ):
                return True

            if (
                span.start_line
                and span.start_line <= codeblock.start_line
                and span.end_line >= codeblock.end_line
            ):
                return True
        return False


class FileContext:

    def __init__(self, repo_path: str, files: List[ContextFile]):
        self._repo_path = repo_path
        self._parser = PythonParser()

        self._files = files
        self._codeblocks = {}
        self._span_index = {}

        # Settings
        self._span_marker = SpanMarker.COMMENT
        self._show_header = True

        for file in files:
            full_file_path = os.path.join(self._repo_path, file.file_path)
            with open(full_file_path, "r") as f:
                content = f.read()

            codeblock = self._parser.parse(content)
            self._codeblocks[file.file_path] = codeblock

    def create_prompt(self):
        file_context_content = ""
        for file in self._files:
            codeblock = self._codeblocks[file.file_path]
            if file.spans:
                content = self.print_by_spans(codeblock, file.spans)
            else:
                content = self._print_block(codeblock)
            file_context_content += f"{file.file_path}\n```\n{content}\n```\n"
        return file_context_content
