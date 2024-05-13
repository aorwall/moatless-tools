import json
import logging
import os
import random
import re
from typing import List, Tuple

from litellm import completion
from llama_index.core import get_tokenizer
from openai import OpenAI

from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_with_blockpath_comments
from moatless.constants import SMART_MODEL
from moatless.prompts import SELECT_BLOCKS_SYSTEM_PROMPT
from moatless.types import Usage, BlockPath, BaseResponse

logger = logging.getLogger(__name__)


JSON_SCHEMA = """{
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "string",
            "description": "Your thoughts on how to approach the task."
        },
        "block_paths": {
            "type": "array",
            "description": "List of block paths that are relevant to the requirement.",
            "items": {
                "type": "string",
             }
         }
    },
    "required": ["thoughts"],
    "additionalProperties": False
}
"""


class CodeSelectorResponse(BaseResponse):
    block_paths: List[BlockPath] = []


class CodeBlockSelector:

    def __init__(self, repo_path: str, max_tokens: int = 2048, model_name: str = None):
        self._model_name = model_name or SMART_MODEL
        self._repo_path = repo_path
        self._tokenizer = get_tokenizer()
        self._max_tokens = max_tokens
        self._parser = PythonParser()
        self._client = OpenAI()

    def select_blocks(self, instructions: str, file_path: str) -> CodeSelectorResponse:
        full_file_path = os.path.join(self._repo_path, file_path)
        with open(full_file_path, "r") as f:
            file_content = f.read()

        tokens = len(self._tokenizer(file_content))
        if tokens < self._max_tokens:
            thoughts = (
                f"File {file_path} has {tokens} tokens which is less than max tokens {self._max_tokens}."
                f" Will not select blocks but return the full file content."
            )
            logger.info(thoughts)
            return CodeSelectorResponse(thoughts=thoughts, block_paths=[])

        codeblock = self._parser.parse(file_content)

        system_prompt = self._system_prompt(codeblock)
        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {"content": instructions, "role": "user"}

        messages = [
            system_message,
            instruction_message,
        ]

        usage_stats = []

        is_retry = False
        while True:

            if self._model_name.startswith("gpt"):
                llm_response = self._client.chat.completions.create(
                    model=self._model_name,
                    temperature=0.01,
                    response_format={"type": "json_object"},
                    max_tokens=1024,
                    # tools=tools,
                    messages=messages,
                )
            else:
                llm_response = completion(
                    model=self._model_name,
                    temperature=0.01,
                    max_tokens=512,
                    stop=None,  # '</selected_blocks>',
                    messages=messages,
                )

            if llm_response.usage:
                usage_stats.append(Usage.parse_obj(llm_response.usage.dict()))

            corrections = ""
            thoughts = None
            block_paths = []

            if llm_response.choices[0].message.content:
                try:
                    selected_block_json = self._handle_response(
                        llm_response.choices[0].message.content
                    )

                    for block_path_str in selected_block_json["block_paths"]:
                        if block_path_str.startswith(
                            "block_path: "
                        ):  # Because GPT 3.5 adds "block_path:" again sometimes...
                            block_path_str = block_path_str[len("block_path: ") :]

                        if block_path_str == "start":
                            block_paths.append(
                                ["__start__"]
                            )  # TODO: Would like to just add [] here and handle it properly...
                            continue

                        block_path = block_path_str.split(".")

                        found_block = codeblock.find_by_path(block_path)
                        if not found_block:
                            found_blocks = codeblock.find_blocks_with_identifier(
                                block_path[-1]
                            )
                            if len(found_blocks) == 0:
                                corrections += f"No block found with block_path `{block_path_str}`. Please provide a correct block_path. "
                            elif len(found_blocks) > 1:
                                found_block_paths = ",".join(
                                    [block.path_string() for block in found_blocks]
                                )
                                corrections += f"Multiple blocks found with block_path `{block_path_str}`: {found_block_paths}. Specify the full block_path to the relevant blocks. "
                            else:
                                found_block = found_blocks[0]

                        if found_block:
                            block_paths.append(found_block.full_path())

                    thoughts = selected_block_json["thoughts"]
                except Exception as e:
                    logger.warning(
                        f"Error parsing message {llm_response.choices[0].message.content}. Error: {e}"
                    )
                    corrections = f"The format of your message is incorrect. {e}"
            else:
                logger.warning("No tool call or message found in response.")
                corrections = "No tool call or message found in response."

            messages.append(
                {
                    "content": llm_response.choices[0].message.content,
                    "role": "assistant",
                }
            )

            if corrections and not is_retry:
                logger.info(f"Retry with corrections: {corrections}")
                messages.append({"content": corrections, "role": "user"})
                is_retry = True
                logger.warning(f"No tasks found in response. Retrying...")
            else:
                return CodeSelectorResponse(
                    thoughts=thoughts, block_paths=block_paths, usage_stats=usage_stats
                )

    def _handle_response(self, content):
        logger.info(f"Got content: {content}")
        if self._model_name.startswith("gpt"):
            return json.loads(content)
        else:
            thoughts, block_paths = self.parse_custom_format(content)
            return {"block_paths": block_paths, "thoughts": thoughts}

    def parse_custom_format(self, text: str) -> Tuple[str, List[str]]:
        parts = text.split("<selected_blocks>")
        pre_text = parts[0].strip() if len(parts) > 0 else ""
        block_paths = self.parse_blocks(text)
        return pre_text, block_paths

    def parse_blocks(self, text: str) -> List[str]:
        # Check for both as might Haiku send both...
        pattern_block_id_attr = re.compile(r'<block id=(["\'])(.*?)\1>', re.DOTALL)
        pattern_block_id_tag = re.compile(r"<block_id>([^<]+)</block_id>", re.DOTALL)

        values = []

        for match in pattern_block_id_attr.findall(text):
            id_value = self._clean_value(match[1])
            values.append(id_value)

        for match in pattern_block_id_tag.findall(text):
            values.append(self._clean_value(match))

        return values

    def _clean_value(self, value: str) -> str:
        """Handle more Haiku weirdness."""
        if ", " in value:
            value = value.replace(", ", ".")

        return value

    def _system_prompt(self, codeblock: CodeBlock):
        content = print_with_blockpath_comments(codeblock, field_name="block_id")

        if self._model_name.startswith("gpt"):
            return f"""# FILE:
```python
{content}
``` 

========================

INSTRUCTIONS:
{SELECT_BLOCKS_SYSTEM_PROMPT}

Respond ONLY in JSON that follows the schema below:"
{JSON_SCHEMA}
"""

        else:
            few_shot = self.few_shot_example(codeblock)
            return f"""<file_content>
{content}
</file_content>

========================

<instructions>
{SELECT_BLOCKS_SYSTEM_PROMPT}

Respond with the following format:
<selected_blocks>
<block id='[Block ID]'>
<block id='[Block ID]'>
</selected_blocks>

</instructions>

{few_shot}
"""

    def few_shot_example(self, codeblock: CodeBlock):
        functions = codeblock.find_blocks_with_type(CodeBlockType.FUNCTION)
        if functions:
            random_block = random.choice(functions)
            label = "function"
        else:
            classes = codeblock.find_blocks_with_type(CodeBlockType.CLASS)
            if classes:
                label = "class"

                random_block = random.choice(classes)
            else:
                return ""

        return f"""<example>
Requirement: Improve the error handling in {random_block.identifier} 

Output:
This requirement must be related to the {label} {random_block.identifier} with the same name. 

<selected_blocks>
<block id='{random_block.full_path()}'>
</selected_blocks>

</example>
"""
