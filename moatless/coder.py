import difflib
import json
import logging
import os
import re
from typing import List, Optional, Union

from litellm import completion
from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.print_block import print_by_block_path
from moatless.codeblocks.parser.python import PythonParser
from moatless.constants import SMART_MODEL
from moatless.prompts import CODER_SYSTEM_PROMPT
from moatless.types import Usage

logger = logging.getLogger(__name__)


class CodePart(BaseModel):
    file_path: Optional[str] = None
    language: Optional[str] = None
    content: str


def extract_response_parts(response: str) -> List[Union[str, CodePart]]:
    """
    This function takes a string containing text and code blocks.
    It returns a list of CodePart and non-code text in the order they appear.

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


    Parameters:s
    text (str): The input string containing code blocks and text

    Returns:
    list: A list containing instances of CodeBlock, FileBlock, and non-code text strings.
    """

    combined_parts = []

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
                        logging.info(f"Found multiple files in preceding text: {all_matches}, will set {file_path}")

        # If there's any non-code preceding text, append it to the parts
        if preceding_text:
            combined_parts.append(preceding_text)

        if match.group("language1") or match.group("code1"):
            language = match.group("language1") or None
            content = match.group("code1").strip()
        else:
            language = match.group("language2").lower()
            content = match.group("code2").strip()

        code_block = CodePart(file_path=file_path, language=language, content=content)

        combined_parts.append(code_block)

        last_end = end

    remaining_text = response[last_end:].strip()
    if remaining_text:
        combined_parts.append(remaining_text)

    return combined_parts


tools = [
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Write code in sthe pecfied block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Your thoughts on how to approach the task."
                    },
                    "change": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string"
                            },
                            "block_path": {
                                "type": "string",
                                "description": "Path to code blocks that should be updated.",
                            },
                            "content": {
                                "type": "string",
                                "descriptions": "Updated content"
                            },
                        },
                        "required": ["file_path", "content"],
                        "additionalProperties": False
                    }
                }
            }
        }
    }
]


def do_diff(file_path: str, original_content: str, updated_content: str) -> Optional[str]:
    return "".join(difflib.unified_diff(
        original_content.strip().splitlines(True),
        updated_content.strip().splitlines(True),
        fromfile=file_path, tofile=file_path, lineterm="\n"))


class CodeChange(BaseModel):
    file_path: str
    block_path: str
    content: str


class WriteCodeFunction(BaseModel):
    thoughts: Optional[str] = None
    change: CodeChange = None


class InvalidCodeBlock(BaseModel):
    block_path: str
    content: str
    message: str


class WriteCodeResponse(BaseModel):
    content: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None


class CoderResponse(BaseModel):
    thoughts: Optional[str] = None
    file_path: str
    diff: Optional[str] = None
    error: Optional[str] = None
    change: Optional[str] = None
    usage_stats: List[Usage] = None


class Coder:

    def __init__(self,
                 repo_path: str,
                 model_name: str = None,
                 log_file: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._parser = PythonParser(apply_gpt_tweaks=True)
        self._log_file = log_file

    def write_code(self, main_objective: str, instructions: str, file_path: str, block_path: str) -> CoderResponse:
        full_file_path = os.path.join(self._repo_path, file_path)

        with open(full_file_path, "r") as file:
            original_content = file.read()

        codeblock = self._parser.parse(original_content)
        block_path = block_path.split(".")

        # TODO: Verify if block paths exists in codeblock

        #block_content = print_by_block_paths(codeblock, block_paths)

        system_prompt = f"""
{CODER_SYSTEM_PROMPT}
"""

        system_message = {"content": system_prompt, "role": "system"}

        instruction_code = print_by_block_path(codeblock, block_path).strip()

        instruction_message = {"role": "user",
                               "content": f"""# Main objective: 
{main_objective}

# Current instructions:

{instructions}

# Existing code:

```python
{instruction_code}
```
"""
}
        messages = [
            system_message,
            instruction_message
        ]

        usage_stats = []
        thoughts = None

        retry = 0
        while retry < 3:
            llm_response = completion(
                model=self._model_name,
                temperature=0.0,
                # tools=tools,
                messages=messages)

            if llm_response.usage:
                usage_stats.append(Usage.parse_obj(llm_response.usage.dict()))

            change = llm_response.choices[0].message.content

            extracted_parts = extract_response_parts(change)

            changes = [part for part in extracted_parts if isinstance(part, CodePart)]

            thoughts = [part for part in extracted_parts if isinstance(part, str)]
            thoughts = "\n".join([thought for thought in thoughts])

            if changes:
                if len(changes) > 1:
                    logger.warning(f"Multiple code blocks found in response, ignoring all but the first one.")

                response = self._write_code(codeblock, block_path, changes[0].content)

                if response.content:
                    diff = do_diff(full_file_path, original_content, response.content)
                    if diff:
                        logger.debug(f"Writing updated content to {full_file_path}")
                        # TODO: Write directly to file??
                        with open(full_file_path, "w") as file:
                            file.write(response.content)
                    else:
                        logger.info("No changes detected.")

                    if not response.error:
                        messages.append(llm_response.choices[0].message.model_dump())
                        self._save_message_log(messages)
                        return CoderResponse(
                            thoughts=thoughts,
                            usage_stats=usage_stats,
                            file_path=file_path,
                            diff=diff,
                            change=response.change
                        )

                change = f"\nHere's the updated block {block_path}:\n```\n{response.content}\n```\n"
                corrections = f"\nThe block {block_path} isn't correct.\n{response.error}\n"

                assistant_message = {"role": "assistant", "content": change}
                messages.append(assistant_message)
                correction_message = {"role": "user", "content": corrections}

                logger.info(f"Ask to the LLM to retry with the correction message: {correction_message}")
                messages.append(correction_message)

                retry += 1

        self._save_message_log(messages)

        return CoderResponse(
            thoughts=thoughts,
            usage_stats=usage_stats,
            file_path=file_path,
            error="Failed to update code blocks."
        )



    def _write_code(self, original_block: CodeBlock, block_path: List[str], change: str, action: Optional[str] = None) -> WriteCodeResponse:

        def respond_with_invalid_block(message: str):
            return WriteCodeResponse(
                change=change,
                error=message,
            )

        try:
            codeblock = self._parser.parse(change)
        except Exception as e:
            logger.error(f"Failed to parse block content: {e}")
            # TODO: Instruct the LLM to fix the block content
            return respond_with_invalid_block(f"Syntex error: {e}")

        if action == "delete":
            logger.info(f"Want to delete block {block_path}, but can't because it's not supported yet")
            # TODO: Delete block
            return WriteCodeResponse(
                content=original_block.to_string(),
                change=None
            )

        if action == "add":
            logger.info(f"Want to create block {block_path}, but can't because it's not supported yet")
            # TODO: Add block
            return WriteCodeResponse(
                content=original_block.to_string(),
                change=None
            )

        changed_block = self.find_by_path_recursive(codeblock, block_path)
        if not changed_block:
            logger.warning(f"Couldn't find expected blockpath {block_path} in content:\n{change}")
            return respond_with_invalid_block("The updated block should not contain multiple blocks.")

        error_blocks = changed_block.find_errors()
        if error_blocks:
            logger.warning(f"Syntax errors found in updated block:\n{change}")
            error_block_report = "\n\n".join([f"```{block.content}```" for block in error_blocks])
            return respond_with_invalid_block(f"There are syntax errors in the updated file:\n\n{error_block_report}. "
                                              f"Correct the errors and try again.")

        if not changed_block.is_complete():
            logger.warning(f"Updated block isn't complete:\n{change}")
            return respond_with_invalid_block("The code is not fully implemented. Write out all code in the code block.")

        original_block.replace_by_path(block_path, changed_block)
        logger.debug(f"Updated block: {block_path}")

        return WriteCodeResponse(
            content=original_block.to_string(),
            change=change
        )

    def find_by_path_recursive(self, codeblock, block_path: List[str]):
        found = codeblock.find_by_path(block_path)
        if not found and len(block_path) > 1:
            return self.find_by_path_recursive(codeblock, block_path[1:])
        return found

    def _handle_tool_call(self, response):
        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            logger.debug(json.dumps(function_args, indent=2))

            if function_name == "write_code":
                if write_code:
                    logger.warning(f"Multiple write_code functions found, ignoring the second one: {function_args}")
                else:
                    write_code = WriteCodeFunction.parse_obj(function_args)
            else:
                logger.warning(f"Unknown function: {function_name}")

    def _save_message_log(self, messages: List[dict]):
        if self._log_file:
            content = ""
            for message in messages:
                if message:
                    content += "\n\n" + ("=" * 80) + "\n\n"

                content += f"{message['role']}:\n {message['content']}\n"

            with open(self._log_file, "a") as file:
                file.write(content)
