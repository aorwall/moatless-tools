import json
import logging
from typing import List

from litellm import completion
from llama_index.core import get_tokenizer

from moatless.coder.add_code import AddCode
from moatless.coder.prompt import (
    create_system_prompt,
    CREATE_PLAN_PROMPT,
)
from moatless.coder.remove_code import RemoveCode
from moatless.coder.update_code import UpdateCode
from moatless.file_context import FileContext
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class Coder:

    def __init__(self, file_context: FileContext):

        self._file_context = file_context
        self._tokenizer = get_tokenizer()

    def run(self, requirement: str, mock_response: List[str] = None):
        file_context_content = self._file_context.create_prompt(
            show_span_ids=True, show_line_numbers=True
        )

        system_prompt = f"""# Instructions
{CREATE_PLAN_PROMPT}

# File context

{file_context_content}

"""

        system_message = {"content": system_prompt, "role": "system"}
        instruction_message = {
            "content": f"# Requirement\n{requirement}",
            "role": "user",
        }

        messages = [
            system_message,
            instruction_message,
        ]

        tools = [
            UpdateCode.openai_tool_spec,
            AddCode.openai_tool_spec,
            RemoveCode.openai_tool_spec,
        ]

        correction = ""

        response = completion(
            model=Settings.coder.planning_model,
            max_tokens=750,
            temperature=0.0,
            tools=tools,
            metadata={
                "generation_name": "coder-plan",
                "trace_name": "coder",
            },
            messages=messages,
            mock_response=mock_response,
        )
        response_message = response.choices[0].message

        # TODO Måste fixas
        return False
