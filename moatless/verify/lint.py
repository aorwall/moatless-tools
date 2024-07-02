import logging
import tempfile
from typing import Optional

from astroid import MANAGER
from pylint.lint import Run
from pylint.testutils import MinimalTestReporter

from moatless.repository import CodeFile
from moatless.types import VerificationError
from moatless.verify.verify import Verifier

logger = logging.getLogger(__name__)


class PylintVerifier(Verifier):

    def __init__(self, repo_dir: str, run_tests: bool = True):
        self.repo_dir = repo_dir
        self.run_tests = run_tests

    def verify(self, file: Optional[CodeFile] = None) -> list[VerificationError]:
        if not file:
            logger.warning("No file to verify")
            return []

        try:
            MANAGER.astroid_cache.clear()
            results = Run(
                [f"{self.repo_dir}/{file.file_path}"],
                exit=False,
                reporter=MinimalTestReporter(),
            )

            for msg in results.linter.reporter.messages:
                logger.debug(f"Message: {msg.msg_id} {msg.msg} {msg.line}")

            return [
                VerificationError(
                    code=msg.msg_id,
                    file_path=msg.file_path.replace(f"{self.repo_dir}/", ""),
                    message=msg.msg,
                    line=msg.line,
                )
                for msg in results.linter.reporter.messages
                if msg.msg_id[0] in ["E", "F"]
            ]
        except Exception as e:
            logger.exception(f"Error running pylint")
            return []
