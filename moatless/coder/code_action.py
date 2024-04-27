import logging
from abc import abstractmethod

from litellm import completion

from moatless.analytics import send_event
from moatless.codeblocks.module import Module
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder.code_utils import extract_response_parts, CodePart, do_diff
from moatless.coder.types import CoderResponse, WriteCodeResult, CodeFunction
from moatless.file_context import FileContext
from moatless.prompts import CODER_SYSTEM_PROMPT
from moatless.settings import Settings

logger = logging.getLogger(__name__)


class CodeAction:

    def __init__(self, file_context: FileContext = None):
        self._file_context = file_context
        self._parser = PythonParser(apply_gpt_tweaks=True)

    def execute(self, task: CodeFunction, mock_response: str = None):
        if self._file_context.is_in_context(task.file_path):
            logger.error(f"File {task.file_path} not found in file context.")
            return CoderResponse(
                file_path=task.file_path,
                error="The provided file isn't found in the file context.",
            )

        original_module = self._file_context.get_module(task.file_path)

        system_prompt = self._system_instructions()
        task_instructions = self._task_instructions(task, original_module)

        file_context_content = self._file_context.create_prompt()

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

        retry = 0
        while retry < 3:
            llm_response = completion(
                model=Settings.coder.coding_model,
                temperature=0.0,
                max_tokens=2000,
                messages=messages,
                metadata={"generation_name": "coder-write-code", "trace_name": "coder"},
                mock_response=mock_response,
            )

            change = llm_response.choices[0].message.content

            extracted_parts = extract_response_parts(change)

            changes = [part for part in extracted_parts if isinstance(part, CodePart)]

            thoughts = [part for part in extracted_parts if isinstance(part, str)]
            thoughts = "\n".join([thought for thought in thoughts])

            if thoughts:
                logger.info(f"Thoughts: {thoughts}")

            if not changes:
                logger.info("No code changes found in response.")
                send_event(
                    event="code_update_failed",
                    properties={"error_type": "no_changes"},
                )

                return False

            if len(changes) > 1:
                logger.warning(
                    f"Multiple code blocks found in response, ignoring all but the first one."
                )

            try:
                updated_module = self._parser.parse(changes[0].content)
            except Exception as e:
                logger.error(f"Failed to parse block content: {e}")
                corrections = f"There was a syntax error in the code block. Please correct it and try again. Error: '{e}'"
                send_event(
                    event="code_update_failed",
                    properties={"error_type": "syntax_error"},
                )
                updated_module = None

            if updated_module:
                original_content = original_module.to_string()

                response = self._execute(task, original_module, updated_module)

                if not response.error:
                    updated_content = original_module.to_string()
                    diff = do_diff(task.file_path, original_content, updated_content)

                    if diff:
                        # TODO: Move this to let the agent decide if content should be saved
                        self._file_context.update_module(task.file_path, updated_module)
                    else:
                        logger.info("No changes detected.")

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

                    task.state = "completed"

                    return CoderResponse(file_path=task.file_path, diff=diff)

                if response.error:
                    send_event(
                        event="code_update_failed",
                        properties={
                            "error_message": response.error,
                            "error_type": response.error_type,
                        },
                    )
                    corrections = f"The code isn't correct.\n{response.error}\n"
                else:
                    send_event(
                        event="code_update_failed",
                        properties={
                            "error_type": "no changes",
                        },
                    )
                    corrections = "No changes detected."

            change = f"```\n{changes[0].content}\n```\n"

            assistant_message = {"role": "assistant", "content": change}
            messages.append(assistant_message)
            correction_message = {"role": "user", "content": corrections}
            messages.append(correction_message)

            logger.info(
                f"Ask to the LLM to retry with the correction message: {correction_message}"
            )

            retry += 1

        return CoderResponse(
            file_path=task.file_path,
            error="Failed to update code blocks.",
        )

    @abstractmethod
    def _execute(
        self, task: CodeFunction, original_module: Module, updated_module: Module
    ) -> WriteCodeResult:
        pass

    def _system_instructions(self):
        return CODER_SYSTEM_PROMPT

    @abstractmethod
    def _task_instructions(self, task: CodeFunction, module: Module) -> str:
        pass


def respond_with_invalid_block(message: str, error_type: str = None):
    return WriteCodeResult(
        change=None,
        error_type=error_type,
        error=message,
    )
