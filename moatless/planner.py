import json
import logging
import os
from typing import List, Optional

import lunary
from litellm import completion
from llama_index.core import get_tokenizer
from openai import BaseModel, OpenAI

from moatless.code_graph import CodeGraph
from moatless.codeblocks.codeblocks import PathTree, CodeBlockType, CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import BlockMarker, print_by_block_paths, print_block
from moatless.constants import SMART_MODEL
from moatless.prompts import CREATE_DEV_PLAN_SYSTEM_PROMPT
from moatless.types import Usage, BlockPath, ContextFile, DevelopmentTask, BaseResponse

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



class PlannerResponse(BaseResponse):
    tasks: List[DevelopmentTask]


class Planner:

    def __init__(self, repo_path: str, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path

        self._client = OpenAI()
        lunary.monitor(self._client)
        self._tokenizer = get_tokenizer()
        self._show_relationships = True
        self._max_file_token_size = 2000
        self._min_tokens_for_planning = 500  # TODO: Find a good number of tokens where the LLM won't be lazy

    def plan_development(self, instructions: str, files: List[ContextFile]) -> PlannerResponse:
        file_context_content = ""

        tokens = 0

        for file in files:
            full_file_path = os.path.join(self._repo_path, file.file_path)
            with open(full_file_path, 'r') as f:
                file_content = f.read()

            tokens += len(self._tokenizer(file_content))

            # TODO: Just for verifying benchmark. Should be moved to a centralized code index!
            code_graph = CodeGraph()

            def add_to_graph(codeblock: CodeBlock):
                code_graph.add_to_graph(file.file_path, codeblock)

            parser = PythonParser(index_callback=add_to_graph)

            codeblock = parser.parse(file_content)

            block_paths = []

            if self._show_relationships:
                for selected_block_path in file.block_paths:
                    block_paths.append(selected_block_path)

                    related_blocks = code_graph.find_relationships(file.file_path, selected_block_path)
                    if related_blocks:
                        for related_block in related_blocks:
                            if related_block.type in [CodeBlockType.CLASS, CodeBlockType.MODULE]:
                                for child in related_block.children:
                                    if not child.is_indexed and child.full_path() not in block_paths:
                                        block_paths.append(child.full_path())

                            elif related_block.full_path() not in block_paths:
                                block_paths.append(related_block.full_path())

            if file.block_paths:
                content = print_by_block_paths(codeblock,
                                               block_paths=block_paths,
                                               # show_types=[CodeBlockType.CLASS, CodeBlockType.FUNCTION],
                                               block_marker=BlockMarker.COMMENT)
                # TODO: Check min / max_tokens and iterate
            else:
                content = print_block(codeblock, block_marker=BlockMarker.COMMENT)

            file_context_content += f"{file.file_path}\n```\n{content}\n```\n"

        if tokens < self._min_tokens_for_planning:
            thought = f"File(s) have less than {self._min_tokens_for_planning} tokens. Will provide the full file."
            return PlannerResponse(
                thoughts=thought,
                tasks=[
                    DevelopmentTask(
                        file_path=file.file_path,
                        instructions=instructions,
                        block_path=[],
                        state="planned",
                        action="update"
                    ) for file in files
                ]
            )

        system_prompt = f"""# FILES:
{file_context_content}

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
        while retry < 2:
            response = self._client.chat.completions.create(
                model=self._model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=messages)

            if response.usage:
                usage_stats.append(Usage.parse_obj(response.usage.dict()))

            if response.choices[0].message.content:
                try:
                    devplan_json = json.loads(response.choices[0].message.content)
                    # TODO: Verify that blocks exists

                    tasks = []
                    for task_json in devplan_json["tasks"]:
                        if task_json.get("block_path"):
                            block_path = task_json["block_path"].split(".")
                        else:
                            block_path = None

                        tasks.append(DevelopmentTask(
                            file_path=task_json["file_path"],
                            instructions=task_json["instructions"],
                            block_path=block_path,
                            action=task_json.get("action", "update"),
                            state="planned"
                        ))

                    return PlannerResponse(
                        thoughts=devplan_json.get("thoughts", None),
                        tasks=tasks,
                        usage_stats=usage_stats
                    )
                except Exception as e:
                    logger.warning(f"Error parsing message {response.choices[0].message.content}. Error {type(e)}: {e} ")
                    correction = f"Error parsing message: {e}. Please respond with JSON that follows the schema below:\n{JSON_SCHEMA}"
            else:
                logger.warning("No message found in response.")
                raise ValueError("No message found in response.")

            messages.append({"content": response.choices[0].message.content,
                             "role": "assistant"})
            messages.append({"content": correction,
                             "role": "user"})
            retry += 1
            logger.warning(f"No tasks found in response. Retrying {retry}/3...")

        # TODO: Return error?
        return PlannerResponse(
            thoughts="No tasks found in response.",
            usage_stats=usage_stats)

    def _file_context_prompt(self, files: List[ContextFile]) -> str:
        file_context_content = ""
        for file in files:
            full_file_path = os.path.join(self._repo_path, file.file_path)
            with open(full_file_path, 'r') as f:
                file_content = f.read()

            codeblock = self._parser.parse(file_content)

            if file.block_paths:
                content = print_by_block_paths(codeblock,
                                               block_paths=file.block_paths,
                                               # show_types=[CodeBlockType.CLASS, CodeBlockType.FUNCTION],
                                               block_marker=BlockMarker.COMMENT)
                # TODO: Check min / max_tokens and iterate
            else:
                content = print_block(codeblock, block_marker=BlockMarker.COMMENT)

            file_context_content += f"{file.file_path}\n```\n{content}\n```\n"
        return file_context_content
