import json
import logging
import os
from typing import List, Optional

from litellm import completion
from llama_index.core import get_tokenizer
from openai import BaseModel, OpenAI

from moatless.codeblocks.codeblocks import PathTree, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import BlockMarker, print_by_block_paths, print_block
from moatless.constants import SMART_MODEL
from moatless.prompts import CREATE_DEV_PLAN_SYSTEM_PROMPT
from moatless.types import Usage, BlockPath

logger = logging.getLogger(__name__)

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

JSON_SCHEMA = """{
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
                    "action": {
                        "type": "string",
                        "description": "If the code block should be added, removed or updated.",
                        "enum": ["add", "remove", "update"]
                    }
                    "instructions": {
                        "type": "string",
                        "descriptions": "Instructions for editing the code block"
                    },
                },
                "required": ["file_path", "instructions"],
                "additionalProperties": False
            }
        }
    },
    "required": ["thoughts", "tasks"],
    "additionalProperties": False
}"""


class DevelopmentTask(BaseModel):
    file_path: str
    instructions: str
    block_path: str
    action: Optional[str] = None


class DevelopmentPlan(BaseModel):
    thoughts: Optional[str] = None
    tasks: List[DevelopmentTask]


class PlannerResponse(BaseModel):
    thoughts: Optional[str] = None
    usage_stats: List[Usage] = None
    tasks: List[DevelopmentTask]


class Planner:

    def __init__(self, repo_path: str, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._parser = PythonParser()
        self._client = OpenAI()
        self._tokenizer = get_tokenizer()
        self._max_file_token_size = 2000

    def plan_development(self, instructions: str, files: List[str], block_paths: List[BlockPath] = None) -> PlannerResponse:
        file_context = self._file_context_prompt(files, block_paths)
        system_prompt = f"""# FILES:
{file_context}

========================

INSTRUCTIONS:
{CREATE_DEV_PLAN_SYSTEM_PROMPT}

Respond ONLY in JSON that follows the schema below:"
{JSON_SCHEMA}
"""

        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}

        messages = [
            system_message,
            instruction_message,
        ]

        usage_stats = []
        retry = 0
        while retry < 3:
            response = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                # tools=tools,
                messages=messages)

            if response.usage:
                usage_stats.append(Usage.parse_obj(response.usage.dict()))

            if response.choices[0].message.content:
                try:
                    devplan_json = json.loads(response.choices[0].message.content)
                    # TODO: Verify that blocks exists
                    devplan = DevelopmentPlan.parse_obj(devplan_json)
                    return PlannerResponse(
                        thoughts=devplan.thoughts,
                        tasks=devplan.tasks,
                        usage_stats=usage_stats
                    )
                except Exception as e:
                    logger.warning(f"Error parsing message {response.choices[0].message.content}. Error: {e}")
            else:
                logger.warning("No tool call or message found in response.")

            messages.append({"content": response.choices[0].message.content,
                             "role": "assistant"})
            messages.append({"content": "You must use the function `create_development_plan` to create a plan with a list of tasks!",
                             "role": "user"})
            retry += 1
            logger.warning(f"No tasks found in response. Retrying {retry}/3...")

        # TODO: Return error?
        return PlannerResponse(
            thoughts="No tasks found in response.",
            usage_stats=usage_stats)

    def _handle_tool_call(self, response):
        if response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls

            for tool_call in tool_calls:
                logger.debug(f"\nExecuting tool call: {tool_call}")
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logger.debug(json.dumps(function_args, indent=2))

                if function_name == "create_development_plan":
                    # TODO: Handle more than one tool call
                    devplan = DevelopmentPlan.parse_obj(function_args)

                else:
                    logger.debug(f"Unknown function: {function_name}")

    def _file_context_prompt(self, files: List[str], block_paths: List[BlockPath] = None) -> str:
        file_context_content = ""
        for file_path in files:
            full_file_path = os.path.join(self._repo_path, file_path)
            with open(full_file_path, 'r') as f:
                file_content = f.read()

            codeblock = self._parser.parse(file_content)

            tokens = len(self._tokenizer(file_content))
            if tokens > self._max_file_token_size and block_paths:  # TODO: Refactor to support block paths per file
                logger.info(f"File {file_path} is too large ({tokens} tokens). Selecting blocks {block_paths}")
                content = print_block(codeblock,
                                      path_tree=PathTree.from_block_paths(block_paths),
                                      show_types=[CodeBlockType.CLASS, CodeBlockType.FUNCTION],
                                      block_marker=BlockMarker.COMMENT)
            else:
                blockpaths = None
                content = print_block(codeblock, blockpaths, block_marker=BlockMarker.COMMENT)

            file_context_content += f"{file_path}\n```\n{content}\n```\n"
        return file_context_content
