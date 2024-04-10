import json
import logging
import os
from typing import List

import anthropic

from litellm import completion, Message
from llama_index.core import get_tokenizer

from moatless.code_index import CodeIndex
from moatless.codeblocks.codeblocks import Span, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_by_line_numbers


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
                    "description": "A semantic similarity search query. Use natural language to describe what you are looking for."
                },
                "class_names": {
                    "type": "array",
                    "description": "Name of the classed to find and get more details about.",
                    "items": {
                        "type": "string"
                    }
                },
                "function_names": {
                    "type": "array",
                    "description": "Names of a functions to find and get more details about.",
                    "items": {
                        "type": "string"
                    }
                },
                "file_names": {
                    "type": "array",
                    "description": "Filter out search on specific file names.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["query"],
        }
    }
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
            }
        }
    }
}

tools = [
    search_function,
    finish_function
]

system_prompt = """<instructions>
Your task is to find the file that needs to be updated based on the request. 

Use the search function to find the file. You can narrow down the search by specifying the class, function,
and file name if you're sure what these should be set to.

When you find the right file, you should use the finish function to select the file.

Think step by step and start by write out your thoughts on how to define the search parameters. 
MAX 50 words. Do not write anything else!
</instructions>
"""


logger = logging.getLogger(__name__)


class Search:

    def __init__(self,
                 code_index: CodeIndex,
                 path: str,
                 model: str = "claude-3-opus-20240229",
                 max_tokens: int = 15000,
                 max_file_tokens: int = 8000):
        self._code_index = code_index
        self._path = path
        self._parser = PythonParser()
        self._max_file_tokens = max_tokens
        self._max_tokens = max_file_tokens
        self._tokenize = get_tokenizer()
        self._model = model

        self._client = anthropic.Anthropic()

    def search(self, request: str):
        calls = 0

        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": request, "role": "user"},
        ]

        while calls < 5:
            calls += 1

            # TODO: Filter out old messages and irrelevant context

            try:
                response = completion(
                    model=self._model,
                    max_tokens=500,
                    temperature=0.0,
                    tools=tools,
                    messages=messages
                )

                response_message = response['choices'][0]['message']
            except Exception as e:
                logger.warning(f"Failed to do request with messages: {messages}. Error {e}")
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

                logger.info(f"{tool_call.function.name}:\n{json.dumps(function_args, indent=2)}")

                if tool_call.function.name == "finish":
                    if "file_path" not in function_args:
                        logger.warning("No file path found in response.")
                        return None

                    # TODO: Handle both aboslute and relative paths...
                    file_path = function_args["file_path"]
                    if file_path.startswith(self._path):
                        return file_path[len(self._path)+1:]
                    else:
                        return file_path

                if "query" in function_args:
                    function_response = self._search(query=function_args["query"],
                                                     file_names=function_args.get("file_names"),
                                                     class_names=function_args.get("class_names"),
                                                     function_names=function_args.get("function_names"))

                    if not function_response:
                        function_response = "No results found."

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": function_response,
                    })
                else:
                    logger.warning(f"Unknown response: {tool_call.function.name}")

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": "Unknown response",
                    })

        logger.warning("Max number of calls reached. Giving up!")
        return None

    def _count_tokens(self, message):
        if isinstance(message, dict):
            return len(self._tokenize(str(message["content"])))
        elif message.content:
            return len(self._tokenize(str(message.content)))
        else:
            return 0  # TODO: Handle tokens in tools

    def _search(self, query: str = None,
                file_names: List[str] = None,
                class_names: List[str] = None,
                function_names: List[str] = None):

        snippets = self._code_index.retriever.retrieve(query)

        blocks = {}

        spans = {}

        sum_tokens = 0

        for snippet in snippets:
            if file_names and not any(file_name in snippet.file_path for file_name in file_names):
                continue

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

                with open(os.path.join(self._path, file_path), "r") as file:
                    content = file.read()

                codeblock = self._parser.parse(content)

                blocks[snippet.file_path] = codeblock

            maching_blocks = codeblock.find_indexed_blocks_by_spans([Span(snippet.start_line, snippet.end_line)])

            if class_names:
                filtered_blocks = []
                for class_name in class_names:
                    filtered_blocks.extend([block for block in maching_blocks if class_name in block.identifier])
                maching_blocks = filtered_blocks

            if function_names:
                filtered_blocks = []
                for function_name in function_names:
                    filtered_blocks.extend([block for block in maching_blocks if function_name in block.identifier])
                maching_blocks = filtered_blocks

            if not maching_blocks:
                continue

            if snippet.file_path not in spans:
                spans[snippet.file_path] = []

            if snippet.tokens + sum_tokens > self._max_file_tokens:
                break

            sum_tokens += snippet.tokens

            spans[snippet.file_path].append(Span(snippet.start_line, snippet.end_line))

        response = ""
        for file_path, spans in spans.items():
            codeblock = blocks[file_path]
            trimmed_content = print_by_line_numbers(codeblock, spans=spans)
            response += f"\n{file_path}\n```python\n{trimmed_content}\n```\n"

        return response


if "main" in __name__:
    search = Search()
    search._search("find the file that contains the function `get_data`")
