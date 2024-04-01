import logging
import os
import re
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET

from litellm import completion, ModelResponse
from llama_index.core import get_tokenizer
from pydantic import BaseModel, Field

from moatless.codeblocks import CodeBlockType, CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print import print_by_line_numbers
from moatless.prompts import SELECT_FILES_SYSTEM_PROMPT
from moatless.constants import CHEAP_MODEL
from moatless.retriever import CodeSnippet
from moatless.types import CodeFile
from moatless.utils.xml import extract_between_tags

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


class SelectFilesResponse(BaseModel):
    thoughts: str = None
    files: List[CodeFile]
    usage_stats: List[dict] = None


class FileSelector:

    def __init__(self,
                 repo_path: str,
                 model_name: str = None,
                 file_context_token_limit: int = 25000,
                 max_files: int = 50,
                 tokenizer=None):
        self._model_name = model_name or CHEAP_MODEL
        self._repo_path = repo_path
        self._file_context_token_limit = file_context_token_limit
        self._max_files = max_files
        self._tokenizer = tokenizer or get_tokenizer()  # TODO: How to tokenize Claude?
        self._parser = PythonParser()

    def select_files(self, query: str, code_snippets: List[CodeSnippet]) -> SelectFilesResponse:

        sum_tokens = 0
        usage_stats = []

        selected_files = []
        selected_not_in_prompt = []
        files_to_read = []

        provided_files = self._group_code_snippets(code_snippets)

        thoughts = None

        retry = 0
        while retry < 3:
            retry += 1
            prompt_files = []

            if files_to_read:
                for file_to_read in files_to_read:
                    with open(os.path.join(self._repo_path, file_to_read), 'r') as f:
                        content = f.read()

                    tokens = len(self._tokenizer(content))
                    if sum_tokens + tokens < self._file_context_token_limit:
                        file = CodeFile(file_path=file_to_read, content=content, is_complete=True)
                        prompt_files.append(file)
                    else:
                        logger.warning(f"File requested to read {file_to_read} is too large to include in the prompt.")
                        # TODO: Do one more round?

                for selected_file in selected_files:
                    if selected_file.file_path not in files_to_read:
                        tokens = len(self._tokenizer(selected_file.content)) if selected_file.content else 0
                        if sum_tokens + tokens < self._file_context_token_limit:
                            prompt_files.append(selected_file)
                    else:
                        selected_not_in_prompt.append(selected_file)
            else:
                for file in provided_files:
                    if files_to_read:
                        if file.file_path not in files_to_read:
                            continue
                        else:
                            with open(os.path.join(self._repo_path, file.file_path), 'r') as f:
                                file.content = f.read()

                    if any([file.file_path == prompt_file.file_path for prompt_file in prompt_files]):
                        # already added to the prompt
                        continue

                    tokens = len(self._tokenizer(file.content)) if file.content else 0
                    if sum_tokens + tokens > self._file_context_token_limit:
                        if len(prompt_files) == len(selected_files):
                            logger.info(f"Selected {len(selected_files)} files out of {len(provided_files)} in the response ({sum_tokens} tokens).")
                            return SelectFilesResponse(
                                files=selected_files,
                                usage_stats=usage_stats
                            )

                        # Remove content when context grows too large
                        file.content = None

                    prompt_files.append(file)
                    sum_tokens += tokens

            logger.info(f"Will check {len(prompt_files)} files out of {len(provided_files)} in total ({sum_tokens} tokens).")

            file_by_idx = {idx: f for idx, f in enumerate(prompt_files)}

            response = completion(
                model=self._model_name,
                temperature=0.0,
                max_tokens=1000,
                messages=self._create_claude_prompt(query, file_by_idx)
            )

            usage_stats.append(response.usage.dict())
            selected_files_response, files_to_read, thoughts = self._parse_claude_response(response, file_by_idx)

            selected_files_response.extend(selected_not_in_prompt)
            selected_not_in_prompt = []

            selected_files = []
            sum_tokens = 0
            for selected_file in selected_files_response:
                tokens = len(self._tokenizer(selected_file.content))
                if sum_tokens + tokens > self._file_context_token_limit:
                    logger.info(f"Selected {len(selected_files)} code snippets out of {len(selected_files_response)} in the response ({sum_tokens} tokens).")

                    if files_to_read:
                        logger.warning(f"Files requested to read but was never read: {files_to_read}")
                        # TODO: Handle this?

                    return SelectFilesResponse(
                        files=selected_files,
                        usage_stats=usage_stats
                    )

                selected_files.append(selected_file)
                sum_tokens += tokens

            if not files_to_read:
                return SelectFilesResponse(
                    files=selected_files,
                    usage_stats=usage_stats
                )

        return SelectFilesResponse(
            thoughts=thoughts,
            files=selected_files,
            usage_stats=usage_stats
        )

    def _code_snippet_to_xml(self, file: CodeFile, idx: int = 0):
        return f"""<document index="{idx}">
<source>{file.file_path}</source>
{file.content or '# ... commented out code ...'}
</document>
"""

    def _code_snippet_to_filename(self, file: CodeFile):
        return f"\n<file_path>{file.file_path}</file_path>"

    def _create_claude_prompt(self, requirement: str, file_by_idx: Dict[int, CodeFile]):
        file_context_content = ""

        for idx, file in file_by_idx.items():
            file_context_content += self._code_snippet_to_xml(file, idx)

        prompt = f"""
<documents>
{file_context_content}        
</documents>

{SELECT_FILES_SYSTEM_PROMPT}
"""

        return [
            {"content": prompt, "role": "system"},
            {"content": f"# Requirement:\n{requirement}", "role": "user"}
        ]

    def _parse_claude_response(self, response: ModelResponse, file_by_idx: Dict[int, CodeFile]) -> Tuple[List[CodeFile], List[str], str]:
        response_message = response.choices[0].message.content
        logger.info(response_message)

        thoughts = ""
        thinking = extract_between_tags("thinking", response_message, strip=True)
        for thought in thinking:
            thoughts += "\n" + thought

        files = extract_between_tags("select_document_index", response_message, strip=True)

        selected_files = []
        for i, idx in enumerate(files):
            logger.info(f"File: {idx}")
            file = file_by_idx.get(int(idx))
            if file is None:
                logger.warning(f"Code snippet with index {idx} not found.")
                continue

            already_selected = any([selected_file.file_path == file.file_path for selected_file in selected_files])
            if already_selected:
                continue

            selected_files.append(file)

        read_document_indices = extract_between_tags("read_document_index", response_message, strip=True)

        files_to_read = []
        for read_document_index in read_document_indices:
            file_to_read = file_by_idx.get(int(read_document_index))
            if not file_to_read:
                continue

            if file_to_read.is_complete:
                logger.warning(f"Tried to read file {file_to_read.file_path} that is already complete.")
            else:
                files_to_read.append(file_to_read.file_path)

        return selected_files, files_to_read, thoughts

    def _group_code_snippets(self, code_snippets: List[CodeSnippet]) -> List[CodeFile]:
        snippets_by_file_path = {}

        file_paths = []

        for code_snippet in code_snippets:
            if code_snippet.file_path not in snippets_by_file_path:
                snippets_by_file_path[code_snippet.file_path] = []
                file_paths.append(code_snippet.file_path)

            snippets_by_file_path[code_snippet.file_path].append(code_snippet)

            if len(file_paths) >= self._max_files:
                logger.info(f"Too many files to process. Limiting to {self._max_files} files.")
                break

        grouped_snippets = []

        for file_path in file_paths:
            full_file_path = os.path.join(self._repo_path, file_path)
            with open(full_file_path, 'r') as f:
                file_content = f.read()

            if file_path not in snippets_by_file_path:
                grouped_snippets.append(CodeFile(file_path=file_path, content=file_content, is_complete=True))
                continue

            try:
                codeblock = self._parser.parse(file_content)
            except Exception as e:
                logger.warning(f"Error parsing file {file_path}: {e}")
                grouped_snippets.append(CodeFile(file_path=file_path, content=file_content, is_complete=True))
                continue

            snippets = snippets_by_file_path[file_path]
            sorted_snippets = sorted(snippets, key=lambda x: x.start_line)

            line_numbers = [(snippet.start_line, snippet.end_line) for snippet in sorted_snippets]

            content = print_by_line_numbers(codeblock, line_numbers)
            grouped_snippets.append(CodeFile(file_path=file_path, content=content, is_complete=codeblock.is_complete()))

        return grouped_snippets
