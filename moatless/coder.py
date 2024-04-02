import difflib
import json
import logging
import os
from typing import List, Optional

from litellm import completion
from pydantic import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.print import print_by_block_path
from moatless.codeblocks.parser.python import PythonParser
from moatless.constants import SMART_MODEL
from moatless.prompts import CODER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
    change: Optional[CodeChange] = None


class CoderResponse(BaseModel):
    thoughts: Optional[str] = None
    file_path: str
    content: Optional[str] = None
    diff: Optional[str] = None
    error: Optional[str] = None
    change: Optional[CodeChange] = None


class Coder:

    def __init__(self, repo_path: str, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._parser = PythonParser(apply_gpt_tweaks=True)

    def write_code(self, main_objective: str, instructions: str, file_path: str, block_path: str) -> CoderResponse:
        full_file_path = os.path.join(self._repo_path, file_path)

        with open(full_file_path, "r") as file:
            original_content = file.read()

        codeblock = self._parser.parse(original_content)
        block_path = block_path.split(".")

        # TODO: Verify if block paths exists in codeblock

        #block_content = print_by_block_paths(codeblock, block_paths)


        system_prompt = f"""ORIGINAL FILE:
{file_path}
```python
{original_content}
```

{CODER_SYSTEM_PROMPT}
"""

        system_message = {"content": system_prompt, "role": "system"}

        instruction_code = print_by_block_path(codeblock, block_path)

        instruction_message = {"role": "user",
                               "content": f"""# Main objective: 
{main_objective}

# Current instructions:
{instructions}

Only update the following code and return it in full:

```python
{instruction_code}
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

            if write_code:
                response = self._write_code(codeblock, write_code.change)

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
                        return CoderResponse(
                            thoughts=write_code.thoughts,
                            file_path=file_path,
                            content=response.content,
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

        return CoderResponse(
            file_path=file_path,
            error="Failed to update code blocks."
        )

    def _write_code(self, original_block: CodeBlock, change: CodeChange) -> WriteCodeResponse:

        def respond_with_invalid_block(message: str):
            return WriteCodeResponse(
                change=change,
                error=message,
            )

        block_path = change.block_path.split(".")
        if not original_block.find_by_path(block_path):
            logger.warning(f"Block path {change.block_path} not found.")
            return respond_with_invalid_block(f"Block path {change.block_path} not found.")

        try:
            codeblock = self._parser.parse(change.content)
        except Exception as e:
            logger.error(f"Failed to parse block content: {e}")
            # TODO: Instruct the LLM to fix the block content
            return respond_with_invalid_block(f"Syntex error: {e}")

        changed_block = self.find_by_path_recursive(codeblock, block_path)
        if not changed_block:
            logger.warning(f"Couldn't find expected blockpath {change.block_path} in content:\n{change.content}")
            return respond_with_invalid_block("The updated block should not contain multiple blocks.")

        error_blocks = changed_block.find_errors()
        if error_blocks:
            logger.warning(f"Syntax errors found in updated block:\n{change.content}")
            error_block_report = "\n\n".join([f"```{block.content}```" for block in error_blocks])
            return respond_with_invalid_block(f"There are syntax errors in the updated file:\n\n{error_block_report}. "
                                              f"Correct the errors and try again.")

        if not changed_block.is_complete():
            logger.warning(f"Updated block isn't complete:\n{change.content}")
            return respond_with_invalid_block("The code is not fully implemented. Write out all code in the code block.")

        original_block.replace_by_path(block_path, changed_block)
        logger.debug(f"Updated block: {change.block_path}")

        return WriteCodeResponse(
            content=original_block.to_string(),
            change=change
        )

    def find_by_path_recursive(self, codeblock, block_path: List[str]):
        found = codeblock.find_by_path(block_path)
        if not found and len(block_path) > 1:
            return self.find_by_path_recursive(codeblock, block_path[1:])
        return found
