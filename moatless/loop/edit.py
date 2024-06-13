import logging
from typing import Optional, Type

from moatless.loop.base import BaseState, Finished
from moatless.loop.prompt import SEARCH_REPLACE_PROMPT
from moatless.repository import CodeFile
from moatless.types import ActionSpec, Reject, Message, ActionRequest, Content
from pydantic import PrivateAttr, Field, BaseModel

logger = logging.getLogger(__name__)


class CodeChange(ActionRequest):
    thoughts: Optional[str] = Field(
        default=None, description="The thoughts on the code change."
    )
    replace: str = Field(..., description="The code to replace the existing code with.")
    rejected: bool = Field(..., description="Whether the code change was rejected.")


class EditCode(BaseState):
    description: str
    file_path: str
    span_id: str
    start_line: int
    end_line: int

    _file: Optional[CodeFile] = PrivateAttr(default=None)
    _code_to_replace: Optional[str] = PrivateAttr(default=None)
    _retry: int = PrivateAttr(default=0)
    _messages: list[Message] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        description: str,
        file_path: str,
        span_id: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ):
        super().__init__(
            include_message_history=True,
            description=description,
            file_path=file_path,
            span_id=span_id,
            start_line=start_line,
            end_line=end_line,
        )

    def init(self):
        self._file = self.file_repo.get_file(self.file_path)
        code_lines = self._file.content.split("\n")
        lines_to_replace = code_lines[self.start_line - 1 : self.end_line]
        self._code_to_replace = "\n".join(lines_to_replace)

    def handle_action(self, content: Content):
        msg_split = content.content.split("<replace>")
        if len(msg_split) == 1:
            if not self._add_prepared_response:
                logger.warning(
                    f"No <replace> tag found in response without prepped tag: {msg_split[0]}"
                )
                response_message = Message(
                    role="user",
                    content="You did not provide any code in the replace tag. If you want to reject the instructions, use the reject function.",
                )
                self._messages.append(response_message)
                return

            replacement_code = msg_split[0]
        else:
            if msg_split[0]:
                thought = msg_split[0]
                self.trajectory.save_thought(thought)

            replacement_code = msg_split[1]

        update_result = self._file.update_content_by_line_numbers(
            self.start_line - 1, self.end_line, replacement_code
        )

        if update_result.diff and update_result.updated:
            logger.info(
                f"Updated file {self.file_path} with diff:\n{update_result.diff}"
            )
            self.finish(message=f"```diff\n{update_result.diff}\n```")
            return

        if self._retry > 2:
            logger.warning(f"Failed after {self._retry} retries. Will reject change.")
            self.finish(message=f"Failed to apply changes")
            return

        if update_result.diff:
            logger.warning(f"Diff was not applied:\n{update_result.diff}")
            response_message = Message(
                role="user",
                content=f"The following diff was not applied:\n {update_result.diff}. \n"
                f"Errors:\n{update_result.error}\n"
                f"Make sure that you return the unchanged code in the replace tag exactly as it is. "
                f"If you want to reject the instructions, use the reject function.",
            )
            self._retry += 1

        else:
            logger.info(f"No changes found in {self.file_path}.")
            response_message = Message(
                role="user",
                content=f"The code in the replace tag is the same as in the search. Use the reject function if you "
                f"can't do any changes and want to reject the instructions.",
            )
            self._retry += 1

        self._messages.append(response_message)

    def finish(self, message: str):
        self.transition_to(Finished(reason=message))

    def system_prompt(self) -> str:
        return SEARCH_REPLACE_PROMPT

    def messages(self) -> list[Message]:
        messages = [
            Message(
                role="user",
                content=f"{self.description}\n\n<search>\n{self._code_to_replace}\n</search>",
            )
        ]
        messages.extend(self._messages)
        return messages

    @property
    def _add_prepared_response(self):
        return self.model.startswith("claude")

    def action_type(self) -> Optional[Type[BaseModel]]:
        return None

    def stop_words(self):
        return ["</replace>"]
