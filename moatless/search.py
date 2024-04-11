import json
import logging
import os
from typing import List

from litellm import completion
from llama_index.core import get_tokenizer

from moatless.code_index import CodeIndex
from moatless.codeblocks.codeblocks import Span, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_by_line_numbers
from moatless.retriever import CodeSnippet

search_function = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Use semantic similarity search to find relevant code snippets. Get more information about specific files, classes and functions by providing them as search parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A semantic similarity search query. Use natural language to describe what you are looking for.",
                },
                "class_names": {
                    "type": "array",
                    "description": "Name of the classed to find and get more details about.",
                    "items": {"type": "string"},
                },
                "function_names": {
                    "type": "array",
                    "description": "Names of a functions to find and get more details about.",
                    "items": {"type": "string"},
                },
                "keywords": {
                    "type": "array",
                    "description": "Keywords that should exist in the code.",
                    "items": {"type": "string"},
                },
                "file_names": {
                    "type": "array",
                    "description": "Filter out search on specific file names.",
                    "items": {"type": "string"},
                },
            },
            "required": ["query"],
        },
    },
}

finish_function = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Use finish and select the relevant file if you're sure you found the right file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file",
                }
            },
        },
    },
}

tools = [search_function, finish_function]

system_prompt = """<instructions>
Your task is to find the file that needs to be updated based on the request. 

Use the search function to find the file. You can narrow down the search by specifying the class, function,
and file name if you're sure what these should be set to. 

If you specify the file name you will see the signatures of all classes and methods. 
If you specify function names you will see the full content of those functions.
  
You can specify more than one file, function and class name at the same time.

When you find the right file, you should use the finish function to select the file.

Think step by step and start by write out your thoughts on how to define the search parameters. 
MAX 50 words. Do not write anything else!
</instructions>
"""


logger = logging.getLogger(__name__)


class Search:

    def __init__(
        self,
        code_index: CodeIndex,
        path: str,
        model: str = "claude-3-opus-20240229",
        log_dir: str = None,
        metadata: dict = None,
        max_tokens: int = 16000,
        max_file_tokens: int = 6000,
    ):
        self._code_index = code_index
        self._path = path
        self._log_dir = log_dir
        self._parser = PythonParser()
        self._max_file_tokens = max_file_tokens
        self._max_tokens = max_tokens
        self._tokenize = get_tokenizer()
        self._model = model
        self._metadata = metadata

    def search(self, request: str):
        calls = 0
        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": request, "role": "user"},
        ]

        while calls < 5:
            calls += 1

            # TODO: Filter out old messages and irrelevant context in a better way
            if len(messages) > 5:
                logger.info(f"Too many messages: {len(messages)}. Removing old messages. ")
                messages = messages[:1] + messages[2:]
                logger.info(f"TNow {len(messages)} messages.")

            try:
                response = completion(
                    model=self._model,
                    max_tokens=500,
                    temperature=0.0,
                    metadata=self._metadata,
                    tools=tools,
                    messages=messages,
                )

                response_message = response["choices"][0]["message"]
            except Exception as e:
                logger.warning(
                    f"Failed to do request with messages: {messages}. Error {e}"
                )
                raise e

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
            else:
                tool_call = None

            messages.append(response_message)

            if response_message.content:
                logger.info(f"thoughts: {response_message.content}")

            if tool_call:
                function_args = json.loads(tool_call.function.arguments)

                logger.info(
                    f"{tool_call.function.name}:\n{json.dumps(function_args, indent=2)}"
                )

                if tool_call.function.name == "finish":
                    if "file_path" not in function_args:
                        logger.warning("No file path found in response.")
                        return None

                    # TODO: Handle both aboslute and relative paths...
                    file_path = function_args["file_path"]
                    if file_path.startswith(self._path):
                        return file_path[len(self._path) + 1 :]
                    else:
                        return file_path

                if "query" in function_args:
                    function_response = self._search(
                        query=function_args["query"],
                        file_names=function_args.get("file_names"),
                        class_names=function_args.get("class_names"),
                        function_names=function_args.get("function_names"),
                        keywords=function_args.get("keywords"),
                    )

                    if not function_response:
                        function_response = "No results found."

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": function_response,
                        }
                    )
                else:
                    logger.warning(f"Unknown response: {tool_call.function.name}")

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": "Unknown response",
                        }
                    )

        logger.warning("Max number of calls reached. Giving up!")
        return None

    def _count_tokens(self, message):
        if isinstance(message, dict):
            return len(self._tokenize(str(message["content"])))
        elif message.content:
            return len(self._tokenize(str(message.content)))
        else:
            return 0  # TODO: Handle tokens in tools

    def _search(
        self,
        query: str = None,
        file_names: List[str] = None,
        class_names: List[str] = None,
        function_names: List[str] = None,
        keywords: List[str] = None,
    ):
        full_query = ""
        if file_names:
            full_query += f"Files: {' '.join(file_names)}\n"

        if class_names:
            full_query += f"Classes: {' '.join(class_names)}\n"

        if function_names:
            full_query += f"Functions: {' '.join(function_names)}\n"

        if keywords:
            full_query += f"Keywords: {' '.join(keywords)}\n"

        full_query += query

        snippets = self._code_index.retriever.retrieve(full_query)

        return self._create_response(snippets, file_names, class_names, function_names)

    def _create_response(
        self,
        snippets: List[CodeSnippet],
        file_names: List[str] = None,
        class_names: List[str] = None,
        function_names: List[str] = None,
        keywords: List[str] = None,
    ):

        # Only show signatures of indexed blocks if no function names have been specified
        only_show_signatures = not file_names and not class_names and not function_names

        # Only show found snippets if no function name
        only_show_found_snippets = not function_names

        blocks = {}
        spans_by_file = {}
        response = ""
        sum_tokens = 0

        for snippet in snippets:
            if file_names and not any(
                file_name.lower() in snippet.file_path.lower()
                for file_name in file_names
            ):
                continue

            if snippet.file_path in blocks:
                codeblock = blocks[snippet.file_path]

                # Already added to spans list
                if not only_show_found_snippets:
                    break
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

            if only_show_found_snippets:
                indexed_blocks = codeblock.find_indexed_blocks_by_spans([Span(snippet.start_line, snippet.end_line)])

                for block in indexed_blocks:
                    if class_names and not any(
                            class_name.lower() in block.path_string().lower()
                            for class_name in class_names
                    ):
                        continue

                    if keywords and not any(
                            keyword.lower() in block.to_string().lower()
                            for keyword in keywords
                    ):
                        continue

                    span = Span(block.start_line, block.end_line)

                    if snippet.file_path in spans_by_file and span in spans_by_file[snippet.file_path]:
                        continue

                    if only_show_signatures:
                        tokens = block.tokens
                    else:
                        tokens = block.sum_tokens()

                    if tokens + sum_tokens > self._max_file_tokens:
                        break

                    if snippet.file_path not in spans_by_file:
                        spans_by_file[snippet.file_path] = []

                    spans_by_file[snippet.file_path].append(span)
                    sum_tokens += tokens
            else:
                indexed_blocks = codeblock.find_indexed_blocks()

                spans = []

                for block in indexed_blocks:
                    span = Span(block.start_line, block.end_line)

                    if span in spans:
                        continue

                    if function_names:
                        if block.type == CodeBlockType.FUNCTION and any(
                            function_name.lower() in block.identifier.lower()
                            for function_name in function_names
                        ):
                            spans.append(span)

                    elif class_names:
                        if block.type == CodeBlockType.CLASS and any(
                            class_name.lower() in block.identifier.lower()
                            for class_name in class_names
                        ):
                            spans.append(span)

                    else:
                        spans.append(span)

                if only_show_signatures:
                    trimmed_content = print_by_line_numbers(
                        codeblock, spans=spans, only_show_signatures=True
                    )
                else:
                    trimmed_content = print_by_line_numbers(codeblock, spans=spans)

                tokens = len(self._tokenize(trimmed_content))
                if tokens + sum_tokens > self._max_file_tokens:
                    break

                logger.info(f"Found {len(spans)} in file {snippet.file_path} with {tokens} ({sum_tokens} tokens.")
                response += (
                    f"\n{snippet.file_path}\n```python\n{trimmed_content}\n```\n"
                )
                sum_tokens += tokens

        if spans_by_file:
            for file_path, spans in spans_by_file.items():
                codeblock = blocks[file_path]
                trimmed_content = print_by_line_numbers(
                    codeblock, only_show_signatures=only_show_signatures, spans=spans
                )
                response += f"\n{file_path}\n```python\n{trimmed_content}\n```\n"

            logger.info(f"Found {len(spans_by_file)} files with {sum_tokens} tokens.")

        return response
