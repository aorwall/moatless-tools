import json
import logging
import os
import time
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
                # "keywords": {
                #    "type": "array",
                #    "description": "Keywords that should exist in the code.",
                #    "items": {"type": "string"},
                # },
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
  
Your objective is to find all relevant functionality. So you can specify more than one file, function and class name at the same time. 

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

        session_log = [
            {"role": "user", "content": request},
        ]

        prompt_tokens = 0
        completion_tokens = 0

        last_conversation = []

        while calls < 5:
            calls += 1

            # TODO: Filter out old messages and irrelevant context in a better way
            if len(messages) > 2:
                logger.info(
                    f"Too many messages: {len(messages)}. Removing old messages. "
                )
                messages = messages[:2] + messages[2:]
                logger.info(f"TNow {len(messages)} messages.")

            try:
                response = completion(
                    model=self._model,
                    max_tokens=750,
                    temperature=0.0,
                    metadata=self._metadata,
                    tools=tools,
                    messages=messages + last_conversation,
                )

                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens

                response_message = response["choices"][0]["message"]
            except Exception as e:
                logger.warning(
                    f"Failed to do request with last_conversation: {last_conversation}. Error {e}"
                )
                session_log.append(
                    {
                        "role": "system",
                        "content": "Failed to do request.",
                        "error": str(e),
                    }
                )
                with open(os.path.join(self._log_dir, f"session.json"), "w") as file:
                    json.dump(session_log, file, indent=2)

                raise e

            tool_call = None
            try:
                if response_message.tool_calls:
                    tool_call = response_message.tool_calls[0]
            except Exception as e:
                logger.warning(
                    f"Failed to parse tool call: {response_message}. Error {e}"
                )
                session_log.append(
                    {
                        "role": "system",
                        "content": "Failed to parse tool call.",
                        "error": str(e),
                    }
                )

            last_conversation.append(response_message)

            if response_message.content:
                logger.info(f"thoughts: {response_message.content}")

            if not tool_call:
                logger.warning("No tool call found in response.")
                session_log.append(
                    {"role": "system", "content": "No tool call found in response."}
                )

                last_conversation.append(
                    {"role": "system", "content": "No tool call found in response."}
                )

                continue

            session_log.append(
                {
                    "role": "system",
                    "tool_call": tool_call.model_dump(),
                    "thoughts": response_message.content,
                }
            )

            try:
                function_args = json.loads(tool_call.function.arguments)

                logger.info(
                    f"{tool_call.function.name}:\n{json.dumps(function_args, indent=2)}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to parse arguments: {tool_call.function.arguments}. Error {e}"
                )
                session_log.append(
                    {
                        "role": "system",
                        "content": "Failed to parse arguments",
                        "error": str(e),
                    }
                )
                last_conversation.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": f"Failed to parse arguments: {tool_call.function.arguments}",
                    }
                )
                continue

            if tool_call.function.name == "finish":
                session_log.append(
                    {
                        "role": "system",
                        "content": "Finished search.",
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                )

                with open(os.path.join(self._log_dir, f"session.json"), "w") as file:
                    json.dump(session_log, file, indent=2)

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
                    session_log=session_log,
                    first_request=calls == 1,
                )

                if not function_response:
                    function_response = "No results found."

                    if "file_names" in function_args:
                        function_response += " Try to remove the file name filter."

                    if "class_names" in function_args:
                        function_response += " Try to remove the class name filter."

                    if "function_names" in function_args:
                        function_response += " Try to remove the function name filter."

                last_conversation.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": function_response,
                    }
                )
            else:
                logger.warning(f"Unknown response: {tool_call.function.name}")

                last_conversation.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": "Unknown response",
                    }
                )

        session_log.append(
            {"role": "system", "content": "Max number of calls reached. Giving up!"}
        )

        with open(os.path.join(self._log_dir, f"session.json"), "w") as file:
            json.dump(session_log, file, indent=2)

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
        session_log: List = None,
        first_request: bool = False,
    ):

        keywords = []

        full_query = ""
        if file_names:
            full_query += f"Files: {' '.join(file_names)}\n"

        if class_names:
            full_query += f"Classes: {' '.join(class_names)}\n"
            keywords.extend(class_names)

        if function_names:
            full_query += f"Functions: {' '.join(function_names)}\n"
            keywords.extend(function_names)

        full_query += query

        snippets = self._code_index.retriever.retrieve(
            full_query, file_names=file_names, keyword_filters=keywords
        )

        response = self._create_response(snippets, session_log=session_log)
        tokens = len(self._tokenize(response))

        if (file_names or keywords) and (first_request and tokens < self._max_tokens):
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

            response += self._create_response(
                filtered_snippets, session_log=session_log, sum_tokens=tokens
            )
            tokens = len(self._tokenize(response))

        logger.info(f"Responding with {tokens} tokens.")

        session_log.append(
            {
                "type": "vector_search",
                "query": full_query,
                "file_names": file_names,
                "tokens": tokens,
                "keywords": keywords,
                "results": len(snippets),
            }
        )

        if not response:
            logger.warning(f"No snippets found for query: {full_query}")
            return None

        return response

    def _create_response(
        self,
        snippets: List[CodeSnippet],
        session_log: List = None,
        sum_tokens: int = 0,
        only_show_signatures: bool = False,
    ):

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
