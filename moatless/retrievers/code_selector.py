import json
import logging
from typing import List, Tuple, Dict

from litellm import completion, ModelResponse
from llama_index.core import get_tokenizer
from pydantic import BaseModel, Field

from moatless.coder.code_writer import FileItem
from moatless.constants import CHEAP_MODEL
from moatless.retriever import CodeSnippet
from moatless.utils.xml import extract_between_tags, contains_tag

select_files = {
    "type": "function",
    "function": {
        "name": "select_files",
        "description": "Select files.",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_files_in_context": {
                    "type": "array",
                    "description": "List of paths to the provided files relevant to the query",
                    "items": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "show_next_page": {
                    "type": "boolean",
                    "description": "Set to true if you want to see more files",
                },
                "expected_files": {
                    "type": "string",
                    "description": "Provide information on what types of file you expect to find.",
                }
            }
        }
    }
}

logger = logging.getLogger(__name__)


class SelectedCodeSnippet(BaseModel):
    code_snippet: CodeSnippet = Field(...)
    instruction: str = None

class SelectedCodeResponse(BaseModel):
    selected_code_snippets: List[SelectedCodeSnippet]
    total_snippets_listed: int
    usage_stats: List[dict]


class CodeSelector:

    def __init__(self, model_name: str = None, file_context_token_limit: int = 25000, tokenizer=None):
        self._model_name = model_name or CHEAP_MODEL
        self._file_context_token_limit = file_context_token_limit
        self._tokenizer = tokenizer or get_tokenizer()  # TODO: How to tokenize Claude?

    def select_code_snippets(self, requirement: str, code_snippets: List[CodeSnippet]) -> SelectedCodeResponse:
        selected_code_snippets = []

        i = 0

        finished = False
        sum_tokens = 0

        usage_stats = []

        while i < len(code_snippets) and not finished:
            context_code_snippets = []
            for selected_code_snippet in selected_code_snippets:
                context_code_snippets.append(selected_code_snippet.code_snippet)

            while i < len(code_snippets):
                code_snippet = code_snippets[i]
                tokens = len(self._tokenizer(code_snippet.content))

                if sum_tokens + tokens > self._file_context_token_limit:
                    if len(context_code_snippets) == len(selected_code_snippets):
                        logger.info(f"Selected {len(selected_code_snippets)} code snippets out of {len(code_snippets)} in the response ({sum_tokens} tokens).")
                        return SelectedCodeResponse(
                            selected_code_snippets=selected_code_snippets,
                            total_snippets_listed=i,
                            usage_stats=usage_stats
                        )

                    break

                context_code_snippets.append(code_snippet)
                sum_tokens += tokens
                i += 1

            logger.info(f"Will check {len(context_code_snippets)} code snippets out of {len(code_snippets)} in total ({sum_tokens} tokens, index {i}).")

            snippet_by_idx = {idx: code_snippet for idx, code_snippet in enumerate(context_code_snippets)}

            response = completion(
                model=self._model_name,
                temperature=0.0,
                max_tokens=1000,
                messages=self._create_claude_prompt(requirement, snippet_by_idx)
            )

            usage_stats.append(response.usage.dict())
            selected_code_snippets_response, finished = self._parse_claude_response(response, snippet_by_idx)

            selected_code_snippets = []
            sum_tokens = 0
            for selected_code_snippet in selected_code_snippets_response:
                tokens = len(self._tokenizer(selected_code_snippet.code_snippet.content))
                if sum_tokens + tokens > self._file_context_token_limit:
                    logger.info(f"Selected {len(selected_code_snippets)} code snippets out of {len(selected_code_snippets_response)} in the response ({sum_tokens} tokens).")
                    return SelectedCodeResponse(
                        selected_code_snippets=selected_code_snippets,
                        total_snippets_listed=i,
                        usage_stats=usage_stats
                    )

                selected_code_snippets.append(selected_code_snippet)
                sum_tokens += tokens

        return SelectedCodeResponse(
            selected_code_snippets=selected_code_snippets,
            total_snippets_listed=i,
            usage_stats=usage_stats
        )

    def _code_snippet_to_xml(self, code_snippet: CodeSnippet, idx: int = 0):
        return f"""<document index="{idx}">
<source>{code_snippet.file_path}</source>
<document_content>{code_snippet.content}</document_content>
</document>
"""

    def _do_claude_request(self, requirement: str, code_snippets: List[CodeSnippet]):
        snippet_by_idx = {idx: code_snippet for idx, code_snippet in enumerate(code_snippets)}

        response = completion(
            model=self._model_name,
            temperature=0.0,
            max_tokens=1000,
            messages=self._create_claude_prompt(requirement, snippet_by_idx)
        )

        return self._parse_claude_response(response, snippet_by_idx)

    def _create_claude_prompt(self, requirement: str, snippet_by_idx: Dict[int, CodeSnippet]):

        file_context_content = ""
        for idx, code_snippet in snippet_by_idx.items():
            file_prompt = self._code_snippet_to_xml(code_snippet, idx)
            file_context_content += file_prompt

        prompt = f"""
<documents>
{file_context_content}        
</documents>

<instructions>
Act as an expert software engineer with superiour python programming skills.
You have been provided with new software requirement and a paged list of files and their contents.
Your task is to select the files that needs to be changed based on the users requirements. 

Respond by listing the document indexes with <document_index></document_index> tags combined with a <instruction> tag 
describing what needs to be changed or checked in the file. You should list the files that are relevant to the users instructions.
List the files by relevance, the first file should be the most relevant and the last file should be the least relevant.

If you believe that there are more files that are relevant to the users requirement you should instruct the system to 
provide more files with the tag <show_next_page> and provide information on what types of file you expect to find.

If you found all relevant files add the tag <finished>.

Do not provide any other information than withing the tags! 

Before answering the question, please think about it step-by-step within <thinking></thinking> tags.
</instructions>"""

        return [
            {"content": prompt, "role": "system"},
            {"content": f"Requirement: {requirement}", "role": "user"}
        ]

    def _parse_claude_response(self, response: ModelResponse, snippet_by_idx: Dict[int, CodeSnippet]) -> Tuple[List[SelectedCodeSnippet], bool]:
        response_message = response.choices[0].message.content
        logger.info(response_message)

        thinking = extract_between_tags("thinking", response_message, strip=True)
        for thought in thinking:
            logger.info(f"Thinking: {thought}")

        files = extract_between_tags("document_index", response_message, strip=True)
        instructions = extract_between_tags("instruction", response_message, strip=True)

        selected_files = []
        for i, idx in enumerate(files):
            logger.info(f"File: {idx}")
            code_snippet = snippet_by_idx.get(int(idx))
            if code_snippet is None:
                logger.warning(f"Code snippet with index {idx} not found.")
                continue

            if len(files) == len(instructions):
                selected_files.append(SelectedCodeSnippet(code_snippet=code_snippet, instruction=instructions[i]))
            else:
                selected_files.append(SelectedCodeSnippet(code_snippet=code_snippet))

        finished = contains_tag("finished", response_message)
        if finished:
            logger.info("Finished")
        else:
            show_next_page = extract_between_tags("show_next_page", response_message, strip=True)
            logger.info(f"Show next page: {show_next_page}")

        return selected_files, finished
