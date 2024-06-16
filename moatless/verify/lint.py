import tempfile

from pydantic import BaseModel
from pylint.lint import Run
from pylint.message import Message, MessageDefinition
from pylint.testutils import MinimalTestReporter


class LintMessage(BaseModel):
    lint_id: str
    message: str
    line: int


def _run_pylint(content: str) -> list[Message]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(content.encode())
        temp_file.flush()
        temp_file_path = temp_file.name

    results = Run([temp_file_path], exit=False, reporter=MinimalTestReporter())
    return results.linter.reporter.messages


def lint_updated_code(
    original_content: str, updated_content: str, language: str = "python"
) -> list[LintMessage]:
    if language != "python":
        raise ValueError("Only python language is supported for linting")

    try:
        original_messages = _run_pylint(original_content)
        updated_messages = _run_pylint(updated_content)

        original_message_set = set((msg.msg_id, msg.msg) for msg in original_messages)
        updated_message_set = set((msg.msg_id, msg.msg) for msg in updated_messages)

        added_messages_set = updated_message_set - original_message_set

        added_messages = [
            LintMessage(lint_id=msg.msg_id, message=msg.msg, line=msg.line)
            for msg in updated_messages
            if (msg.msg_id, msg.msg) in added_messages_set
        ]

        return added_messages
    except Exception as e:
        raise e
