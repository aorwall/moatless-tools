import json
import logging
import os
from typing import List, Optional

from litellm import completion
from openai import BaseModel

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.constants import SMART_MODEL
from moatless.prompts import CREATE_DEV_PLAN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DevelopmentTask(BaseModel):
    file_path: str
    instructions: str
    block_path: str


class DevelopmentPlan(BaseModel):
    thoughts: Optional[str] = None
    tasks: List[DevelopmentTask]



class Planner:

    def __init__(self, repo_path: str, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._parser = PythonParser()

    def plan_development(self, instructions: str, files: List[str]) -> DevelopmentPlan:

        file_context = self._file_context_prompt(files)
        system_prompt = f"# FILES:\n{file_context}\n\n========================\n\nINSTRUCTIONS:\n{CREATE_DEV_PLAN_SYSTEM_PROMPT}"
        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_development_plan",
                    "description": "Create a plan for how to update the code base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thoughts": {
                                "type": "string",
                                "description": "Your thoughts on how to approach the task."
                            },
                            "tasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {
                                            "type": "string",
                                            "description": "Path to the file that should be updated."
                                        },
                                        "block_path": {
                                            "type": "string",
                                            "description": "Path to code block that should be updated."
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

        messages = [
            system_message,
            instruction_message,
        ]

        retry = 0
        while retry < 3:
            response = completion(
                model=self._model_name,
                temperature=0.0,
                tools=tools,
                messages=messages)

            try:
                if response.choices[0].message.content:
                    logger.info(f"Got unexpected message: {response.choices[0].message.content}")

                if response.choices[0].message.tool_calls:
                    tool_calls = response.choices[0].message.tool_calls

                    for tool_call in tool_calls:
                        logger.debug(f"\nExecuting tool call: {tool_call}")
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        logger.debug(json.dumps(function_args, indent=2))

                        if function_name == "create_development_plan":
                            # TODO: Handle more than one tool call
                            return DevelopmentPlan.parse_obj(function_args)
                        else:
                            logger.debug(f"Unknown function: {function_name}")
                else:
                    logger.warning("No tool calls found in response.")
            except Exception as e:
                logger.error(f"Error processing response: {e}")

            messages.append({"content": response.choices[0].message.content,
                             "role": "assistant"})
            messages.append({"content": "You must use the function `create_development_plan` to create a plan with a list of tasks!",
                             "role": "user"})
            retry += 1
            logger.warning(f"No tasks found in response. Retrying {retry}/3...")

        # TODO: Return error?
        return DevelopmentPlan()

    def _file_context_prompt(self, files: List[str]) -> str:
        file_context_content = ""
        for file_path in files:
            full_file_path = os.path.join(self._repo_path, file_path)
            with open(full_file_path, 'r') as f:
                file_content = f.read()

            codeblock = self._parser.parse(file_content)
            content = self._to_code_block_string(codeblock)
            file_context_content += f"{file_path}\n```\n{content}\n```\n"
        return file_context_content

    def _to_code_block_string(self, codeblock: CodeBlock) -> str:
        contents = ""

        if not codeblock.parent:
            contents += f"# block_path: start"

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
