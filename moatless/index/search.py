import logging
import logging
import os
from typing import List

from pydantic import Field

from moatless.analytics import send_event
from moatless.codeblocks.print_block import print_by_line_numbers
from moatless.coder.types import Function, FunctionResponse
from moatless.retriever import CodeSnippet
from moatless.types import Span

logger = logging.getLogger(__name__)


class Search(Function):
    """
    Semantic similarity search to find relevant code snippets. Get more
    information about specific files, classes and functions by providing
    them as search parameters.
    """

    query: str = Field(
        ...,
        description="A semantic similarity search query. Use natural language to describe what you are looking for.",
    )
    class_names: List[str] = Field(
        [], description="Name of the classed to find and get more details about."
    )
    function_names: List[str] = Field(
        [], description="Names of a functions to find and get more details about."
    )
    file_names: List[str] = Field(
        [], description="Filter out search on specific file names."
    )
    # keywords: List[str] = Field([], description="Keywords that should exist in the code.")
    file_names: List[str] = Field(
        [], description="Filter out search on specific file names."
    )

    def call(self) -> FunctionResponse:
        keywords = []

        full_query = ""
        if self.file_names:
            full_query += f"Files: {' '.join(self.file_names)}\n"

        if self.class_names:
            full_query += f"Classes: {' '.join(self.class_names)}\n"
            keywords.extend(self.class_names)

        if self.function_names:
            full_query += f"Functions: {' '.join(self.function_names)}\n"
            keywords.extend(self.function_names)

        if self.query:
            full_query += self.query

        snippets = self._code_index.retriever.retrieve(
            full_query, file_names=self.file_names, keyword_filters=keywords
        )

        response = self._create_response(snippets)
        tokens = len(self._tokenize(response))

        if (self.file_names or keywords) and (
            self.first_request and tokens < self._max_tokens
        ):
            extra_snippets = []
            if keywords:
                extra_snippets = self._code_index.retriever.retrieve(
                    full_query, keyword_filters=keywords, top_k=250
                )

            if not extra_snippets:
                extra_snippets = self._code_index.retriever.retrieve(
                    full_query, top_k=250
                )

            filtered_snippets = []
            for extra_snippet in extra_snippets:
                if not any(
                    extra_snippet.file_path == snippet.file_path for snippet in snippets
                ):
                    filtered_snippets.append(extra_snippet)

            logger.info(f"Found {len(filtered_snippets)} extra snippets on file names.")

            response += self._create_response(filtered_snippets, sum_tokens=tokens)
            tokens = len(self._tokenize(response))

        logger.info(f"Responding with {tokens} tokens.")

        if not response:
            function_response = "No results found."

            if "file_names" in self.file_names:
                function_response += " Try to remove the file name filter."

            if "class_names" in self.class_names:
                function_response += " Try to remove the class name filter."

            if "function_names" in self.class_names:
                function_response += " Try to remove the function name filter."

        send_event(
            "search",
            {
                "type": "vector_search",
                "query": full_query,
                "file_names": self.file_names,
                "tokens": tokens,
                "keywords": keywords,
                "results": len(snippets),
            },
        )

        if not response:
            logger.warning(f"No snippets found for query: {full_query}")
            return None

        return FunctionResponse(message=response)

    def _create_response(
        self,
        snippets: List[CodeSnippet],
        session_log: List = None,
        sum_tokens: int = 0,
        only_show_signatures: bool = False,
    ) -> str:

        blocks = {}
        spans_by_file = {}
        response = ""

        only_show_signatures = (
            len(snippets) > 50
        )  # TODO: Do a smarter solution to determine the number of tokens to show in each snippet

        for snippet in snippets:
            if snippet.file_path in blocks:
                codeblock = blocks[snippet.file_path]
            else:
                if os.path.exists(snippet.file_path):
                    file_path = snippet.file_path
                else:
                    file_path = os.path.join(self._path, snippet.file_path)
                    if not os.path.exists(file_path):
                        logger.warning(f"File not found: {file_path}")
                        continue

                with open(file_path, "r") as file:
                    content = file.read()

                codeblock = self._parser.parse(content)

                blocks[snippet.file_path] = codeblock

            # If class names or functions names is specified just find those blocks directly
            # TODO: Do BM25 on keywords?

            tokens = 0
            if only_show_signatures:
                indexed_blocks = codeblock.find_indexed_blocks_by_spans(
                    [Span(snippet.start_line, snippet.end_line)]
                )
                for block in indexed_blocks:
                    tokens += block.tokens
            else:
                tokens = snippet.tokens

            span = Span(snippet.start_line, snippet.end_line)

            if (
                snippet.file_path in spans_by_file
                and span in spans_by_file[snippet.file_path]
            ):
                continue

            if tokens is None:
                continue

            if tokens + sum_tokens > self._max_file_tokens:
                break

            if snippet.file_path not in spans_by_file:
                spans_by_file[snippet.file_path] = []

            spans_by_file[snippet.file_path].append(span)
            sum_tokens += tokens

        if spans_by_file:
            for file_path, spans in spans_by_file.items():
                codeblock = blocks[file_path]
                trimmed_content = print_by_line_numbers(
                    codeblock, only_show_signatures=only_show_signatures, spans=spans
                ).strip()

                response += f"\n{file_path}\n```python\n{trimmed_content}\n```\n"

                # TODO: Try one run handling empty content = len(self._tokenize(trimmed_content))

        session_log.append({"type": "create_response", "spans_by_file": spans_by_file})

        return response
