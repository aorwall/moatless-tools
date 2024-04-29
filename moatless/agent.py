import json
import logging
import uuid
from typing import List

from litellm import completion
from llama_index.core import get_tokenizer

from moatless.coder.add_code import AddCode, AddCodeAction
from moatless.coder.prompt import (
    CREATE_PLAN_PROMPT,
)
from moatless.coder.remove_code import RemoveCode, RemoveCodeAction
from moatless.coder.types import FunctionResponse
from moatless.coder.update_code import UpdateCode, UpdateCodeAction
from moatless.file_context import FileContext
from moatless.session import Session
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class Coder:

    def __init__(self, file_context: FileContext, max_retries: int = 3):
        self._file_context = file_context
        self._tokenizer = get_tokenizer()

        self._trace_id = uuid.uuid4().hex
        self._update_action = UpdateCodeAction(file_context=self._file_context, trace_id=self._trace_id)
        self._add_action = AddCodeAction(file_context=self._file_context, trace_id=self._trace_id)
        self._remove_action = RemoveCodeAction(file_context=self._file_context)
        self._max_retries = max_retries

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

        retries = 0

        while retries < self._max_retries:
            response = completion(
                model=Settings.coder.planning_model,
                max_tokens=750,
                temperature=0.0,
                tools=tools,
                metadata={
                    "generation_name": "coder-plan",
                    "trace_name": "coder",
                    "trace_id": self._trace_id,
                    "session_id": Session.session_id,
                    "tags": Session.tags,
                },
                messages=messages,
            )
            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.content:
                logger.info(f"Response message: {response_message.content}")

            error_messages = []
            if hasattr(response_message, "tool_calls"):
                for tool_call in response_message.tool_calls:
                    logger.info(f"Execute action: {tool_call.function.name}")

                    if tool_call.function.name == UpdateCode.name:
                        task = UpdateCode.model_validate_json(tool_call.function.arguments)
                        response = self._update_action.execute(task)
                    elif tool_call.function.name == AddCode.name:
                        task = AddCode.model_validate_json(tool_call.function.arguments)
                        response = self._add_action.execute(task)
                    elif tool_call.function.name == RemoveCode.name:
                        task = RemoveCode.model_validate_json(tool_call.function.arguments)
                        response = self._remove_action.execute(task)
                    else:
                        logger.warning(f"Unknown function: {tool_call.function.name}")
                        response = FunctionResponse(
                            error=f"Unknown function: {tool_call.function.name}"
                        )

                    if response.error:
                        logger.info(f"Function error response: {response.error}")
                        error_messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": response.error,
                            }
                        )
            else:
                error_messages = [
                    {
                        "role": "user",
                        "content": "You must use one of the functions.",
                    }
                ]

            if error_messages:
                messages.append(error_messages)
                retries += 1
            else:
                return True

        return False
