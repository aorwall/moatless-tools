import logging
import uuid
from abc import abstractmethod

from litellm import completion

from moatless.analytics import send_event
from moatless.codeblocks.module import Module
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_utils import extract_response_parts, CodePart, do_diff
from moatless.coder.types import FunctionResponse, CodeFunction, Function
from moatless.file_context import FileContext
from moatless.prompts import CODER_SYSTEM_PROMPT
from moatless.session import Session
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class CodeAgentFunction(Function):

    def _system_instructions(self):
        return CODER_SYSTEM_PROMPT


class CodeAction:

    def __init__(
        self,
        file_context: FileContext = None,
        trace_id: str = None,
        max_retries: int = 2,
    ):
        self._file_context = file_context or Session.file_context
        self._trace_id = trace_id or uuid.uuid4().hex
        self._parser = PythonParser(apply_gpt_tweaks=True)
        self._max_retries = max_retries

    def execute(
        self, task: CodeFunction, mock_response: str = None
    ) -> FunctionResponse:
        original_file = self._file_context.get_file(task.file_path)
        if not original_file:
            logger.warning(f"File {task.file_path} not found in file context.")
            return FunctionResponse(
                file_path=task.file_path,
                error=f"{task.file_path} was not found in the file context.",
            )

        original_module = original_file.module

        system_prompt = self._system_instructions()
        task_instructions = self._task_instructions(task, original_module)

        file_context_content = self._file_context.create_prompt(
            show_line_numbers=False, exclude_comments=True
        )

        system_message = {
            "role": "system",
            "content": f"""{system_prompt}

# File context:
{file_context_content}
""",
        }

        instruction_message = {
            "role": "user",
            "content": task_instructions,
        }
        messages = [system_message, instruction_message]

        retries = 0

        while retries < self._max_retries:
            rejection_reason = None

            llm_response = completion(
                model=Settings.coder.coding_model,
                temperature=0.0,
                max_tokens=2000,
                messages=messages,
                metadata={
                    "generation_name": "coder-write-code",
                    "trace_name": "coder",
                    "trace_id": self._trace_id,
                    "session_id": Session.session_id,
                    "tags": Session.tags,
                },
                mock_response=mock_response,
            )
            choice = llm_response.choices[0]
            messages.append(choice.message.dict())

            extracted_parts = extract_response_parts(choice.message.content)

            changes = [part for part in extracted_parts if isinstance(part, CodePart)]

            thoughts = [part for part in extracted_parts if isinstance(part, str)]
            thoughts = "\n".join([thought for thought in thoughts])

            if thoughts:
                logger.info(f"Thoughts: {thoughts}")

            if not changes:
                if choice.finish_reason == "length":
                    logger.warning(f"No changed found, probably exeeded token limit.")
                    send_event(
                        event="code_update_failed",
                        properties={
                            "error_type": "token_limit_exceeded",
                            "retries": retries,
                        },
                    )

                    # TODO: Handle in a more graceful way than aborting the flow
                    # To not spend to many tokens...
                    raise Exception(f"Token limit exceeded. ")
                else:
                    logger.info("No code changes found in response.")
                    send_event(
                        event="code_update_failed",
                        properties={"error_type": "no_changes", "retries": retries},
                    )
                    return FunctionResponse(
                        error=f"No code changes found in the updated code. {thoughts}"
                    )

            if len(changes) > 1:
                logger.warning(
                    f"Multiple code blocks found in response, ignoring all but the first one."
                )

            try:
                updated_module = self._parser.parse(changes[0].content)
            except Exception as e:
                logger.error(f"Failed to parse block content: {e}")
                rejection_reason = f"There was a syntax error in the code block. Please correct it and try again. Error: '{e}'"
                send_event(
                    event="code_update_rejected",
                    properties={
                        "error_type": "syntax_error",
                        "rejection_reason": rejection_reason,
                        "retries": retries,
                    },
                )
                updated_module = None

            if updated_module:
                original_content = original_module.to_string()

                response = self._execute(task, original_module, updated_module)

                if response.error:
                    rejection_reason = response.error
                    send_event(
                        event="code_update_rejected",
                        properties={
                            "rejection_reason": rejection_reason,
                            "error_type": response.error_type,
                            "retries": retries,
                        },
                    )
                else:
                    updated_content = original_module.to_string()
                    diff = do_diff(task.file_path, original_content, updated_content)

                    if diff:
                        # TODO: Move this to let the agent decide if content should be saved
                        self._file_context.save_file(task.file_path)
                        logger.info(f"Code updated and saved successfully.")
                    else:
                        logger.warning("No changes detected.")

                    diff_lines = diff.split("\n")
                    added_lines = [
                        line
                        for line in diff_lines
                        if line.startswith("+") and not line.startswith("+++")
                    ]
                    removed_lines = [
                        line
                        for line in diff_lines
                        if line.startswith("-") and not line.startswith("---")
                    ]

                    send_event(
                        event="updated_code",
                        properties={
                            "added_lines": len(added_lines),
                            "removed_lines": len(removed_lines),
                            "file": task.file_path,
                            "span_id": (task.span_id if task.span_id else None),
                        },
                    )

                    return FunctionResponse(message=diff)

            if rejection_reason:
                logger.warning(f"Code update rejected: {rejection_reason}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Sorry, I couldn't update the code. {rejection_reason}",
                    }
                )
                retries += 1
            else:
                break

        return FunctionResponse(
            error=f"Failed to update code. No rejection reason provided."
        )

    @abstractmethod
    def _execute(
        self, task: CodeFunction, original_module: Module, updated_module: Module
    ) -> FunctionResponse:
        pass

    def _system_instructions(self):
        return CODER_SYSTEM_PROMPT

    @abstractmethod
    def _task_instructions(self, task: CodeFunction, module: Module) -> str:
        pass


def respond_with_invalid_block(message: str, error_type: str = None):
    return FunctionResponse(
        change=None,
        error_type=error_type,
        error=message,
    )
