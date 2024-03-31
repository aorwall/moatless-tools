import json
import logging
from typing import List

from litellm import completion
from openai import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_writer import CodeWriter, FileItem, WriteCodeRequest
from moatless.coder.ghostcoder import extract_code_blocks
from moatless.coder.prompts import CODER_SYSTEM_PROMPT, SELECT_BLOCKS_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


select_code = {
                "type": "function",
                "function": {
                    "name": "select_code_blocks",
                    "description": "Select code blocks or files to edit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_code_blocks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string"
                                        },
                                        "block_path": {
                                            "type": "string",
                                            "description": "Path to the code block in the file. Set null to include the whole file"
                                        },
                                        "instructions": {
                                            "type": "string",
                                            "descriptions": "Instructions for editing the code block"
                                        },
                                    },
                                    "required": ["file_path", "instructions"],
                                    "additionalProperties": False
                                }
                            }
                        }
                    }
                }
            }


class SelectedCodeBlock(BaseModel):
    file_path: str
    instructions: str
    block_paths: List[str]


class SelectedCodeBlocks(BaseModel):
    selected_code_blocks: List[SelectedCodeBlock]


class Ghostcoder:

    def __init__(self,
                 main_objective: str,
                 context_files: list[FileItem],
                 model_name: str = "gpt-4-0125-preview"):
        self._model_name = model_name
        self._code_writer = CodeWriter()
        self._parser = PythonParser()

        self._file_context = {}
        for context_file in context_files:
            self._file_context[context_file.file_path] = context_file

        self._main_objective = main_objective

    def run(self):
        selected_code_blocks = self.select_blocks(self._main_objective)
        for selected_code_block in selected_code_blocks.selected_code_blocks:
            self.write_code(selected_code_block, selected_code_block.instructions)

    def select_blocks(self, instructions: str) -> SelectedCodeBlocks:
        system_message = {"content": SELECT_BLOCKS_SYSTEM_PROMPT, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}
        file_context_message = {"content": self._file_context_prompt(), "role": "user"}

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "select_code_blocks",
                    "description": "Select code blocks or files to edit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_code_blocks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string"
                                        },
                                        "block_paths": {
                                            "type": "array",
                                            "description": "Path to code blocks that should be updated.",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "instructions": {
                                            "type": "string",
                                            "descriptions": "Instructions for editing the code block"
                                        },
                                    },
                                    "required": ["file_path", "instructions"],
                                    "additionalProperties": False
                                }
                            }
                        }

                    }
                }
            }
        ]

        response = completion(
            model=self._model_name,
            temperature=0.0,
            tools=tools,
            messages=[
                system_message,
                instruction_message,
                file_context_message
            ])

        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            print(f"\nExecuting tool call: {tool_call}")
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(json.dumps(function_args, indent=2))
            return SelectedCodeBlocks.parse_obj(function_args)

    def write_code(self, selected_code: SelectedCodeBlock, instructions: str):
        codeblock = self._parser.parse(self._file_context[selected_code.file_path].content)

        block_content = self._to_context_string(codeblock, selected_code.block_paths[0].split("."))

        system_message = {"content": CODER_SYSTEM_PROMPT, "role": "system"}
        instruction_message = {"content": f"# Main objective: \n{instructions}\n\n"
                                          f"# Current instructions: {selected_code.instructions}"
                                          f"# Update code:\n{selected_code.file_path}\n"
                                          f"```python\n{block_content}\n```\n",
                               "role": "user"}

        response = completion(
            model=self._model_name,
            temperature=0.0,
            messages=[
                system_message,
                instruction_message
            ])

        print(response.choices[0].message.content)

        updated_files = extract_code_blocks(response.choices[0].message.content)

        #write_request = WriteCodeRequest(
        #    updated_files=updated_files,
        #    file_context=context_files
        #)

        #response = self._code_writer.write_code(write_request)
        return response

    def _file_context_prompt(self) -> str:
        file_context_content = "Here's some files that might be relevant:"
        parser = PythonParser()

        for context_file in self._file_context.values():
            codeblock = parser.parse(context_file.content)
            content = self._to_code_block_string(codeblock, show_block_path=True)
            print(content)
            file_context_content += (f"{context_file.file_path}\n"
                                     f"```{context_file.language}\n"
                                     f"{content}\n"
                                     f"```\n")

        return file_context_content

    def _to_code_block_string(self, codeblock: CodeBlock, show_block_path: bool = False) -> str:
        contents = ""

        if not codeblock.parent:
            contents += f"# block_path: root"

        if codeblock.pre_lines:
            contents += "\n" * (codeblock.pre_lines - 1)

            if codeblock.type in [CodeBlockType.FUNCTION, CodeBlockType.CLASS, CodeBlockType.CONSTRUCTOR]:
                contents += f"\n{codeblock.indentation}# block_path: {codeblock.path_string()}"

            for line in codeblock.content_lines:
                if line:
                    contents += "\n" + codeblock.indentation + line
                else:
                    contents += "\n"
        else:
            contents += codeblock.pre_code + codeblock.content

        for i, child in enumerate(codeblock.children):
            contents += self._to_code_block_string(child)

        return contents

    def _to_context_string(self, codeblock: CodeBlock, block_path: List[str]) -> str:
        contents = ""

        if codeblock.pre_lines:
            contents += "\n" * (codeblock.pre_lines - 1)
            for line in codeblock.content_lines:
                if line:
                    contents += "\n" + codeblock.indentation + line
                else:
                    contents += "\n"
        else:
            contents += codeblock.pre_code + codeblock.content

        has_outcommented_code = False
        for i, child in enumerate(codeblock.children):
            if not block_path and child.type not in [CodeBlockType.CLASS, CodeBlockType.FUNCTION, CodeBlockType.CONSTRUCTOR]:
                if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                    contents += child.create_commented_out_block("... other code").to_string()
                    has_outcommented_code = False
                contents += self._to_context_string(codeblock=child, block_path=block_path)
            elif block_path and block_path[0] == child.identifier:
                if has_outcommented_code and child.type not in [CodeBlockType.COMMENT, CodeBlockType.COMMENTED_OUT_CODE]:
                    contents += child.create_commented_out_block("... other code").to_string()
                    has_outcommented_code = False
                contents += self._to_context_string(codeblock=child, block_path=block_path[1:])
            else:
                has_outcommented_code = True

        return contents

    def _contains_block_paths(self, codeblock: CodeBlock, block_paths: List[List[str]]):
        return [block_path for block_path in block_paths
                if block_path[:len(codeblock.full_path())] == codeblock.full_path()]
