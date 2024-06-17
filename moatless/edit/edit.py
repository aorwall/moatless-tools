import logging
from typing import Optional, Type

from pydantic import PrivateAttr, Field, BaseModel

from moatless.state import AgenticState, Finished
from moatless.repository import CodeFile
from moatless.types import (
    Message,
    ActionRequest,
    ActionResponse,
    Content,
    AssistantMessage,
    UserMessage,
)
from moatless.verify.lint import lint_updated_code

logger = logging.getLogger(__name__)

ROLE_PROMPT = "You are autonomous AI assisistant with superior programming skills."

MAIN_OBJECTIVE_PROMPT = "The main objective is to solve a bigger task specfied by the user, this is wrapped in a <main_objective> tag."

SEARCH_REPLACE_PROMPT = """Your task is to solve a smaller task within the main objective. This task is wrapped in a <task> tag.

The surrounding code context is wrapped in a <file_context> tag.

The code to that should be modified is wrapped in a <search> tag, like this:
<search>
{{CODE}}
</search>

Your task is to update the code inside the <search> tags based on the current task.

When updating the code, please adhere to the following important rules:
- Fully implement the requested change, but do not make any other changes that were not directly asked for
- Do not add any comments describing your changes 
- Indentation and formatting should be the same in the replace code as in the search code
- Ensure the modified code is complete - do not leave any TODOs, placeholder, or missing pieces
- Keep any existing placeholder comments in the <search> block (e.g. # ... other code) - do not remove or implement them

After updating the code, please format your response like this:

<replace>
put the updated code here
</replace>

ONLY return the code that was inside the original <search> tags, but with the requested modifications made. 
Do not include any of the surrounding code.

If all code in the search tag should be removed you can return an empty <replace> tag like this:
<replace>
</replace>

If you can't do any changes and want to reject the instructions return the rejection reason wrapped in a <reject> tag, like this:
<reject>
{{REASON}}
</reject>

Here is an example of what the user's request might look like:

<search>
from flask import Flask 
</search>

And here is how you should format your response:

<replace>
import math
from flask import Flask
</replace>

Here's an example of a rejection response:

<search>
import math
from flask import Flask 
</search>

Remember, only put the updated version of the code from inside the <search> tags in your response, wrapped in <replace>
tags. DO NOT include any other surrounding code than the code in the <search> tag!
"""


class CodeChange(ActionRequest):
    thoughts: Optional[str] = Field(
        default=None, description="The thoughts on the code change."
    )
    replace: str = Field(..., description="The code to replace the existing code with.")
    rejected: bool = Field(..., description="Whether the code change was rejected.")


class EditCode(AgenticState):
    instructions: str
    file_path: str
    span_id: str
    start_line: int
    end_line: int

    show_initial_message: bool = True
    lint_updated_code: bool = False

    _file: Optional[CodeFile] = PrivateAttr(default=None)
    _code_to_replace: Optional[str] = PrivateAttr(default=None)
    _retry: int = PrivateAttr(default=0)
    _messages: list[Message] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        instructions: str,
        file_path: str,
        span_id: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        show_initial_message: bool = True,
        max_iterations: int = 4,
        lint_updated_code: bool = True,
        **data,
    ):
        super().__init__(
            include_message_history=True,
            show_initial_message=show_initial_message,
            max_iterations=max_iterations,
            lint_updated_code=lint_updated_code,
            instructions=instructions,
            file_path=file_path,
            span_id=span_id,
            start_line=start_line,
            end_line=end_line,
            **data,
        )

    def init(self):
        self._file = self.file_context.get_file(self.file_path)
        code_lines = self._file.content.split("\n")
        lines_to_replace = code_lines[self.start_line - 1 : self.end_line]
        self._code_to_replace = "\n".join(lines_to_replace)

    def handle_action(self, content: Content) -> ActionResponse:
        self._messages.append(AssistantMessage(content=content.content))

        if "<reject>" in content.content:
            rejection_message = content.content.split("<reject>")[1].split("</reject>")[
                0
            ]
            return ActionResponse.transition(
                "reject",
                output={"message": rejection_message},
            )

        msg_split = content.content.split("<replace>")
        if len(msg_split) == 1:
            if not self._add_prepared_response:
                logger.warning(
                    f"No <replace> tag found in response without prepped tag: {msg_split[0]}"
                )
                return ActionResponse.retry(
                    "You did not provide any code in the replace tag. If you want to reject the instructions, use the reject function."
                )

            replacement_code = msg_split[0]
        else:
            if msg_split[0]:
                thought = msg_split[0]
                logger.info(f"Thoughts: {thought}")

            replacement_code = msg_split[1]

        update_result = self._file.update_content_by_line_numbers(
            self.start_line - 1, self.end_line, replacement_code
        )

        if update_result.diff and update_result.updated:
            logger.info(
                f"Updated file {self.file_path} with diff:\n{update_result.diff}"
            )

            message = f"Applied the change to {self.file_path}."
            lint_messages = []

            if self.lint_updated_code:
                original_file = self.file_repo.get_file(
                    self.file_path, from_origin=True
                )

                lint_messages = lint_updated_code(
                    language=self._file.module.language,
                    original_content=original_file.content,
                    updated_content=self._file.content,
                )

            return ActionResponse.transition(
                "finish",
                output={
                    "message": message,
                    "diff": update_result.diff,
                    "lint_messages": lint_messages,
                },
            )

        if self._retry > 2:
            logger.warning(f"Failed after {self._retry} retries. Will reject change.")
            return ActionResponse.transition(
                "reject", output={"message": "Failed to apply changes"}
            )

        if update_result.diff:
            logger.warning(f"Diff was not applied:\n{update_result.diff}")
            response_message = (
                f"The following diff was not applied:\n {update_result.diff}. \n"
                f"Errors:\n{update_result.error}\n"
                f"Make sure that you return the unchanged code in the replace tag exactly as it is. "
                f"If you want to reject the instructions, use the reject function."
            )

            self._retry += 1

        else:
            logger.info(f"No changes found in {self.file_path}.")
            response_message = (
                f"The code in the replace tag is the same as in the search. Use the reject function if you "
                f"can't do any changes and want to reject the instructions."
            )

            self._retry += 1

        return ActionResponse.retry(response_message)

    @classmethod
    def required_fields(cls) -> set[str]:
        return {"instructions", "file_path", "span_id", "start_line", "end_line"}

    def finish(self, message: str):
        self.transition_to(Finished(message=message))

    def system_prompt(self) -> str:
        system_prompt = ROLE_PROMPT

        if self.show_initial_message:
            system_prompt += "\n\n"
            system_prompt += MAIN_OBJECTIVE_PROMPT

        system_prompt += "\n\n"
        system_prompt += SEARCH_REPLACE_PROMPT

        return system_prompt

    def messages(self) -> list[Message]:
        file_context_str = self.file_context.create_prompt(
            show_line_numbers=False,
            show_span_ids=False,
            exclude_comments=False,
            show_outcommented_code=True,
            outcomment_code_comment="... other code",
        )

        content = ""
        if self.show_initial_message:
            content = f"<main_objective>\n{self.loop.trajectory.initial_message}\n</main_objective>\n\n"

        content += f"<instructions>\n{self.instructions}\n</instructions>\n"
        content += f"<file_context>\n{file_context_str}\n</file_context>\n"
        content += f"<search>\n{self._code_to_replace}\n</search>"

        messages = [UserMessage(content=content)]

        messages.extend(self.retry_messages())

        if self._add_prepared_response:
            messages.append(AssistantMessage(content="<replace>"))

        return messages

    @property
    def _add_prepared_response(self):
        return self.model.startswith("claude")

    def action_type(self) -> Optional[Type[BaseModel]]:
        return None

    def stop_words(self):
        return ["</replace>"]
