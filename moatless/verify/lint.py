import logging
import tempfile

from astroid import MANAGER
from pylint.lint import Run
from pylint.message import Message
from pylint.testutils import MinimalTestReporter

from moatless.verify.types import VerificationError

logger = logging.getLogger(__name__)


def _run_pylint(content: str) -> list[Message]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(content.encode())
        temp_file.flush()
        temp_file_path = temp_file.name

    results = Run([temp_file_path], exit=False, reporter=MinimalTestReporter())
    return results.linter.reporter.messages


def run_pylint(repo_path: str, file_path: str) -> list[VerificationError]:
    try:
        MANAGER.astroid_cache.clear()
        results = Run(
            [f"{repo_path}/{file_path}"], exit=False, reporter=MinimalTestReporter()
        )

        for msg in results.linter.reporter.messages:
            logger.debug(f"Message: {msg.msg_id} {msg.msg} {msg.line}")

        return [
            VerificationError(
                code=msg.msg_id,
                file_path=file_path.replace(f"{repo_path}/", ""),
                message=msg.msg,
                line=msg.line,
            )
            for msg in results.linter.reporter.messages
            if msg.msg_id[0] in ["E", "F"]
        ]
    except Exception as e:
        logger.exception(f"Error running pylint")
        return []


def lint_updated_code(
    file_path: str, original_content: str, updated_content: str
) -> list[VerificationError]:
    try:
        original_messages = _run_pylint(original_content)
        updated_messages = _run_pylint(updated_content)

        original_message_set = set((msg.msg_id, msg.msg) for msg in original_messages)
        updated_message_set = set((msg.msg_id, msg.msg) for msg in updated_messages)

        added_messages_set = updated_message_set - original_message_set

        added_messages = [
            VerificationError(
                code=msg.msg_id, file_path=file_path, message=msg.msg, line=msg.line
            )
            for msg in updated_messages
            if (msg.msg_id, msg.msg) in added_messages_set
        ]

        return added_messages
    except Exception as e:
        raise e
