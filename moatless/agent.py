import logging
import uuid
from typing import List

from litellm import completion
from litellm.utils import FunctionCall

from moatless.coder.add_code import AddCodeBlock, AddCodeAction
from moatless.coder.remove_code import RemoveCode, RemoveCodeAction
from moatless.coder.types import FunctionResponse
from moatless.coder.update_code import UpdateCode
from moatless.prompts import AGENT_SYSTEM_PROMPT
from moatless.search import Search
from moatless.session import Session
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class Finish(FunctionCall):
    """
    Finish the coding task and submit the solution.
    """


class MoatlessAgent:

    def __init__(self, max_iterations: int = 10):
        self._trace_id = uuid.uuid4().hex
        self._add_action = AddCodeAction(trace_id=self._trace_id)
        self._remove_action = RemoveCodeAction()
        self._max_iterations = max_iterations

    def run(self, requirement: str, mock_response: List[str] = None):
        system_message = {"content": AGENT_SYSTEM_PROMPT, "role": "system"}
        instruction_message = {
            "content": f"# Requirement\n{requirement}",
            "role": "user",
        }

        messages = [
            system_message,
            instruction_message,
        ]

        tools = [
            Search.openai_tool_spec,
            UpdateCode.openai_tool_spec,
            AddCodeBlock.openai_tool_spec,
            RemoveCode.openai_tool_spec,
            Finish.openai_tool_spec,
        ]

        iterations = 0

        while iterations < self._max_iterations:
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

                    if tool_call.function.name == Search.name:
                        func = Search.model_validate_json(tool_call.function.arguments)
                        response = func.execute()
                    elif tool_call.function.name == Finish.name:
                        return True
                    elif tool_call.function.name == UpdateCode.name:
                        task = UpdateCode.model_validate_json(
                            tool_call.function.arguments
                        )
                        response = self._update_action.execute(task)
                    elif tool_call.function.name == AddCodeBlock.name:
                        task = AddCodeBlock.model_validate_json(
                            tool_call.function.arguments
                        )
                        response = self._add_action.execute(task)
                    elif tool_call.function.name == RemoveCode.name:
                        task = RemoveCode.model_validate_json(
                            tool_call.function.arguments
                        )
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
                        logger.info(f"Function response: {response.diff}")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_call.function.name,
                                "content": response.content,
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

            iterations += 1

        return False
