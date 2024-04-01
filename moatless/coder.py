import difflib
import json
import logging
import os
from typing import List, Optional

from litellm import completion
from pydantic import BaseModel

from moatless.codeblocks import CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.prompts import CODER_SYSTEM_PROMPT
from moatless.constants import SMART_MODEL

logger = logging.getLogger(__name__)


tools = [
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Write code in specfied blocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "items": {
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
    changes: List[CodeChange]


class InvalidCodeBlock(BaseModel):
    block_path: str
    content: str
    message: str


class WriteCodeResponse(BaseModel):
    content: str = None
    error: str = None
    invalid_blocks: List[InvalidCodeBlock] = None


class CoderResponse(BaseModel):
    file_path: str
    content: str = None
    diff: str = None
    error: str = None


class Coder:

    def __init__(self, repo_path: str, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._parser = PythonParser(apply_gpt_tweaks=True)

    def write_code(self, main_objective: str, instructions: str, file_path: str, block_paths: List[str]) -> CoderResponse:
        full_file_path = os.path.join(self._repo_path, file_path)

        with open(full_file_path, "r") as file:
            original_content = file.read()

        codeblock = self._parser.parse(original_content)

        block_paths = ",".join(block_paths)
        # TODO: Verify if block paths exists in codeblock

        #block_content = print_by_block_paths(codeblock, block_paths)

        system_message = {"content": CODER_SYSTEM_PROMPT, "role": "system"}

        instruction_message = {"role": "user",
                               "content": f"""# Main objective: 
{main_objective}

# Current instructions:
{instructions}

# Code blocks to update
Only update the following blocks {block_paths}

# Existing code:

{file_path}
```python
{original_content}
```
"""
}
        messages = [
            system_message,
            instruction_message
        ]

        retry = 0
        while retry < 3:
            response = completion(
                model=self._model_name,
                temperature=0.0,
                tools=tools,
                messages=messages)

            write_code = None

            tool_calls = response.choices[0].message.tool_calls
            for tool_call in tool_calls:
                logger.debug(f"\nExecuting tool call: {tool_call}")
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logger.debug(json.dumps(function_args, indent=2))

                if function_name == "write_code":
                    if write_code:
                        logger.warning("Multiple write_code functions found. Extending previous write_code function.")
                        new_write_code = WriteCodeFunction.parse_obj(function_args)
                        write_code.changes.extend(new_write_code.changes)
                    else:
                        write_code = WriteCodeFunction.parse_obj(function_args)
                else:
                    logger.debug(f"Unknown function: {function_name}")

            if write_code:
                response = self._write_code(codeblock, write_code)

                if response.content:
                    diff = do_diff(full_file_path, original_content, response.content)
                    if diff:
                        logger.debug(f"Diff:\n{diff}")
                    else:
                        logger.debug("No changes detected.")

                    # TODO: Write directly to file??
                    with open(full_file_path, "w") as file:
                        file.write(response.content)

                    if not response.invalid_blocks:
                        return CoderResponse(
                            file_path=file_path,
                            content=response.content,
                            diff=diff
                        )

                changes = ""
                corrections = ""
                for block in response.invalid_blocks:
                    changes += f"\nHere's the updated block {block.block_path}:\n```\n{block.content}\n```\n"
                    corrections += f"\nThe block {block.block_path} isn't correct.\n{block.message}\n"

                assistant_message = {"role": "assistant", "content": changes}
                messages.append(assistant_message)
                correction_message = {"role": "user", "content": corrections}
                messages.append(correction_message)

                retry += 1

        return CoderResponse(
            file_path=file_path,
            error="Failed to update code blocks."
        )

    def _write_code(self, codeblock: CodeBlock, write_code: WriteCodeFunction) -> WriteCodeResponse:
        invalid_blocks = []

        for change in write_code.changes:

            def add_invalid_block(message: str):
                invalid_blocks.append(InvalidCodeBlock(
                    block_path=change.block_path,
                    content=change.content,
                    message=message
                ))

            block_path = change.block_path.split(".")
            if not codeblock.find_by_path(block_path):
                add_invalid_block(f"Block path {change.block_path} not found.")
                continue

            try:
                changed_block = self._parser.parse(change.content)
            except Exception as e:
                logger.error(f"Failed to parse block content: {e}")
                # TODO: Instruct the LLM to fix the block content
                return WriteCodeResponse(
                    error=f"Syntex error: {e}"
                )

            if len(changed_block.children) > 1:
                add_invalid_block("The updated block should not contain multiple blocks.")
                continue

            error_blocks = changed_block.find_errors()
            if error_blocks:
                error_block_report = "\n\n".join([f"```{block.content}```" for block in error_blocks])
                add_invalid_block(f"There are syntax errors in the updated file:\n\n{error_block_report}. "
                                  f"Correct the errors and try again.")
                continue

            if not changed_block.is_complete():
                add_invalid_block("The code is not fully implemented. Write out all code in the code block.")
                continue

            codeblock.replace_by_path(block_path, changed_block.children[0])
            logger.debug(f"Updated block: {change.block_path}")

        return WriteCodeResponse(
            content=codeblock.to_string(),
            invalid_blocks=invalid_blocks
        )
