import logging

from astroid import MANAGER, nodes
from pylint.lint import Run
from pylint.testutils import MinimalTestReporter
from astroid.exceptions import AstroidError

from moatless.repository import CodeFile
from moatless.schema import VerificationError
from moatless.verify.verify import Verifier
import builtins
import types

logger = logging.getLogger(__name__)


class PylintVerifier(Verifier):
    def __init__(self, repo_dir: str, run_tests: bool = True):
        self.repo_dir = repo_dir
        self.run_tests = run_tests

    def verify(
        self, file: CodeFile | None = None, retry: bool = False
    ) -> list[VerificationError]:
        if not file:
            logger.warning("No file to verify")
            return []

        try:
            # So this is some LLM generated code to try to fix unclear pylint errors...
            MANAGER.astroid_cache.clear()
            MANAGER.astroid_cache["builtins"] = MANAGER.ast_from_module(builtins)
            generator_node = nodes.ClassDef(
                name="generator",
                lineno=0,
                col_offset=0,
                end_lineno=0,
                end_col_offset=0,
                parent=MANAGER.astroid_cache["builtins"],
            )
            MANAGER.astroid_cache["builtins"].locals["generator"] = [generator_node]

            results = Run(
                [f"{self.repo_dir}/{file.file_path}"],
                exit=False,
                reporter=MinimalTestReporter(),
            )

            for msg in results.linter.reporter.messages:
                logger.debug(f"Message: {msg.msg_id} {msg.msg} {msg.line}")
                if msg.msg_id[0] in ["F"] and not retry:
                    logger.warning(
                        f"Lint error: {msg.msg_id} {msg.msg} {msg.line}. Try again"
                    )
                    return self.verify(file, retry=True)

            return [
                VerificationError(
                    code=msg.msg_id,
                    file_path=msg.path.replace(f"{self.repo_dir}/", ""),
                    message=msg.msg,
                    line=msg.line,
                )
                for msg in results.linter.reporter.messages
                if msg.msg_id[0] in ["E"]
            ]
        except AstroidError:
            logger.warning(
                f"AstroidError occurred while linting {file.file_path}. Skipping this file."
            )
            return []
        except Exception:
            logger.exception("Error running pylint")
            return []
