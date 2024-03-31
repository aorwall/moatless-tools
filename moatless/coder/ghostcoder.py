import json
import logging
import os
import re
from typing import List, Union

from litellm import completion
from openai import BaseModel
from pydantic import Field

from moatless.coder.code_writer import CodeWriter, FileItem, WriteCodeRequest
from moatless.coder.prompts import CODER_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def extract_code_blocks(response: str) -> List[FileItem]:
    """
    This function takes a string containing text and code blocks.
    It returns a list of FileItem objects, each representing a code block.

    The function can parse two types of code blocks:

    1) Backtick code blocks with optional file path:
    F/path/to/file
    ```LANGUAGE
    code here
    ```

    2) Square-bracketed code blocks with optional file path:
    /path/to/file
    [LANGUAGE]
    code here
    [/LANGUAGE]

    Returns:
    list: A list of extracted code blocks
    """

    file_items = []

    # Normalize line breaks
    response = response.replace("\r\n", "\n").replace("\r", "\n")

    # Regex pattern to match code blocks
    block_pattern = re.compile(
        r"```(?P<language1>\w*)\n(?P<code1>.*?)\n```|"  # for backtick code blocks
        r"\[(?P<language2>\w+)\]\n(?P<code2>.*?)\n\[/\3\]",  # for square-bracketed code blocks
        re.DOTALL
    )

    # Define pattern to find files mentioned with backticks
    file_pattern = re.compile(r"`([\w/]+\.\w{1,4})`")

    # Pattern to check if the filename stands alone on the last line
    standalone_file_pattern = re.compile(r'^(?:"|`)?(?P<filename>[\w\s\-./\\]+\.\w{1,4})(?:"|`)?$', re.IGNORECASE)

    last_end = 0

    for match in block_pattern.finditer(response):
        start, end = match.span()

        preceding_text = response[last_end:start].strip()
        preceding_text_lines = preceding_text.split("\n")

        file_path = None

        non_empty_lines = [line for line in preceding_text_lines if line.strip()]
        if non_empty_lines:
            last_line = non_empty_lines[-1].strip()

            filename_match = standalone_file_pattern.match(last_line)
            if filename_match:
                file_path = filename_match.group("filename")
                # Remove the standalone filename from the preceding text
                idx = preceding_text_lines.index(last_line)
                preceding_text_lines = preceding_text_lines[:idx]
                preceding_text = "\n".join(preceding_text_lines).strip()

            # If not found, then check for filenames in backticks
            if not file_path:
                all_matches = file_pattern.findall(last_line)
                if all_matches:
                    file_path = all_matches[-1]  # Taking the last match from backticks
                    if len(all_matches) > 1:
                        logger.info(f"Found multiple files in preceding text: {all_matches}, will set {file_path}")

        if match.group("language1") or match.group("code1"):
            language = match.group("language1") or None
            content = match.group("code1").strip()
        else:
            language = match.group("language2").lower()
            content = match.group("code2").strip()

        code_block = FileItem(file_path=file_path, language=language, content=content)
        file_items.append(code_block)

        last_end = end

    return file_items


class Ghostcoder:

    def __init__(self, model_name: str = "gpt-4-0125-preview"):
        self._model_name = model_name
        self._code_writer = CodeWriter()

    def write_code(self, context_files: list[FileItem], instructions: str):
        system_message = {"content": CODER_SYSTEM_PROMPT, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}

        file_context_content = "Here's some files that might be relevant:\n\n"
        for context_file in context_files:
            file_context_content += (f"{context_file.file_path}\n"
                                     f"```{context_file.language}\n"
                                     f"{context_file.content}\n"
                                     f"```\n")

        file_context_message = {"content": file_context_content, "role": "user"}

        response = completion(
            model=self._model_name,
            temperature=0.0,
            messages=[
                system_message,
                instruction_message,
                file_context_message
            ])

        print(response.choices[0].message.content)

        updated_files = extract_code_blocks(response.choices[0].message.content)

        write_request = WriteCodeRequest(
            updated_files=updated_files,
            file_context=context_files
        )

        response = self._code_writer.write_code(write_request)
        return response


    def _create_file_context_message(self, context_files: list[FileItem]) -> dict:
        file_context_content = "Here's some files that might be relevant:\n\n"
        for context_file in context_files:
            file_context_content += (f"{context_file.file_path}\n"
                                     f"```{context_file.language}\n"
                                     f"{context_file.content}\n"
                                     f"```\n")

        return {"content": file_context_content, "role": "user"}

    def _context_blocks(self, context_files: list[FileItem]) -> str:
        file_context_content = "Here's some files that might be relevant:\n\n"
        for context_file in context_files:
            file_context_content += (f"{context_file.file_path}\n"
                                     f"```{context_file.language}\n"
                                     f"{context_file.content}\n"
                                     f"```\n")
        return file_context_content