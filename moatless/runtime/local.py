import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from swebench.harness.constants import SWEbenchInstance, NON_TEST_EXTS
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, MAP_REPO_VERSION_TO_SPECS

from moatless.exceptions import RuntimeError
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.storage.base import BaseStorage
from moatless.testing.python.parser_registry import parse_log
from moatless.testing.schema import TestResult

logger = logging.getLogger(__name__)


class SweBenchLocalEnvironment(RuntimeEnvironment):
    """Environment implementation for running swebench tests and evaluations."""

    def __init__(
        self,
        repo_path: Path,
        swebench_instance: SWEbenchInstance,
        storage: BaseStorage,
    ):
        logger.info(f"Creating LocalEnvironment instance. ID: {swebench_instance['instance_id']}")
        self.repo_path = repo_path
        self.swebench_instance = swebench_instance
        self.instance_id = swebench_instance["instance_id"]
        self.storage = storage
        self.test_spec = make_test_spec(self.swebench_instance)
        self._install_task = None  # Will hold the installation process task
        self._skip_conda_activate = os.getenv("SKIP_CONDA_ACTIVATE", "false").lower() == "true"

        specs = MAP_REPO_VERSION_TO_SPECS.get(self.swebench_instance["repo"], {}).get(
            self.swebench_instance["version"], {}
        )

        self._install_command = specs.get("install")

        if self.swebench_instance["repo"] == "sphinx-doc/sphinx":
            self._install_after_patch = True
        else:
            self._install_after_patch = False

        self._install_task = asyncio.create_task(self._run_async_installation(self._install_command))

    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        """Run tests with an optional patch and specific test files."""
        # Wait for installation to complete if it's running
        await self._wait_for_install()

        logger.info(f"Starting test {test_files}")

        # Reset repository to clean state
        if not await self._reset_repository():
            raise RuntimeError("Failed to reset repository")

        # Apply patch if provided
        if patch:
            if not await self._apply_patch(patch):
                raise RuntimeError("Failed to apply patch")

        if self._install_after_patch and self._install_command:
            await self._execute_command(self._install_command)

        test_results = []
        if test_files:  # Check if test_files is not None
            test_spec = self.test_spec

            # Run test files one by one to be sure to know which file the test failed on
            for test_file in test_files:
                test_command = self._test_script([test_file])
                logger.info(f"Test command: {test_command}")

                stdout, return_code = await self._execute_command(test_command)

                logger.info(f"Return code: {return_code}")
                if return_code != 0:
                    logger.warning(f"Test output: {stdout}")
                else:
                    logger.debug(f"Test output: {stdout}")

                if self.storage:
                    trajectory_key = self.storage.get_trajectory_path()
                    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_path = f"{trajectory_key}/logs/{datetime_str}_test_run.log"
                    await self.storage.write_raw(log_path, stdout)
                    logger.info(f"Test log saved to {log_path}")

                testbed_results = parse_log(stdout, test_spec.repo)

                for result in testbed_results:
                    if not result.file_path:
                        result.file_path = test_file
                test_results.extend(testbed_results)

        # Reset again after testing
        if not await self._reset_repository():
            raise RuntimeError("Failed to reset repository")

        return test_results

    async def swebench_evaluate(self, evaluation_key: str, patch: str | None = None) -> dict:
        """
        Run evaluation with the provided patch.

        This method applies the patch and runs the evaluation script to check if
        the patch resolves the issue as expected.

        Args:
            patch: The patch to evaluate

        Returns:
            EvaluationResult or None if evaluation fails
        """
        if not patch:
            report = {
                self.instance_id: {
                    "patch_is_None": patch is None,
                    "patch_exists": False,
                    "patch_successfully_applied": False,
                    "resolved": False,
                }
            }
            await self.storage.write(f"{evaluation_key}/report", report)
            return report

        await self._wait_for_install()

        if not patch.endswith("\n"):
            patch += "\n"

        await self.storage.write_raw(f"{evaluation_key}/patch.diff", patch)

        if not await self._apply_patch(patch):
            report = {
                self.instance_id: {
                    "patch_is_None": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "resolved": False,
                }
            }
            await self.storage.write(f"{evaluation_key}/report.json", report)
            return report

        eval_file = Path("/tmp/eval.sh")
        eval_file.write_text(self.test_spec.eval_script)

        await self.storage.write_raw(f"{evaluation_key}/eval.sh", eval_file.read_text())

        logger.debug(f"Executing command: {eval_file} in {self.repo_path}")
        process = await asyncio.create_subprocess_shell(
            f"/bin/bash {eval_file}",
            cwd=str(self.repo_path),
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # redirect stderr to stdout
        )

        stdout, _ = await process.communicate()
        test_output = stdout.decode()
        test_output_path = "/tmp/test_output.log"
        with open(test_output_path, "w") as f:
            f.write(test_output)

        await self.storage.write_raw(f"{evaluation_key}/test_output.txt", test_output)

        report = get_eval_report(
            test_spec=self.test_spec,
            prediction={"instance_id": self.instance_id, "model_patch": patch},
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        await self.storage.write(f"{evaluation_key}/report.json", report)

        return report

    async def _run_async_installation(self, install_command: str | None = None) -> bool:
        """Run installation asynchronously with the improved script approach."""
        if not install_command:
            logger.info("No install command specified in the test spec")
            return True

        if self._install_after_patch:
            logger.info("Skipping installation because it's done after patch")
            return True

        if os.getenv("SKIP_INSTALL", "false").lower() == "true":
            logger.info("Skipping installation")
            return True

        logger.info(f"Running async installation with command: {install_command}")

        stdout, return_code = await self._execute_command(install_command)
        logger.info(f"Installation output: {stdout} {return_code}")
        return return_code == 0

    async def _wait_for_install(self):
        """Wait for the installation process to complete if it's running."""
        if self._install_task is not None and not self._install_task.done():
            logger.info("Waiting for installation process to complete...")
            try:
                await self._install_task
            except Exception as e:
                logger.error(f"Installation process failed: {str(e)}")
                raise RuntimeError("Installation process failed") from e

    async def _execute_command(self, command: str, cwd: Path = None) -> Tuple[str, int]:
        """Execute a shell command and return combined output and return code."""

        # Prepend conda activation to the command using . instead of source
        conda_activate = ". /opt/miniconda3/bin/activate && conda activate testbed && "
        if not self._skip_conda_activate:
            command = conda_activate + command

        bash_command = f'bash -l -c "{command}"'

        logger.debug(f"Executing command: {bash_command} in {cwd}")
        process = await asyncio.create_subprocess_shell(
            bash_command,
            cwd=str(self.repo_path),
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # redirect stderr to stdout
        )

        stdout, _ = await process.communicate()
        return stdout.decode(), process.returncode or 0

    async def _apply_patch(self, patch: str) -> bool:
        """Apply a git patch to the repository."""
        logger.debug("Applying patch to local repository")
        await self._reset_repository()

        # Save patch to a temporary file
        patch_file = self.repo_path / "temp_patch.diff"
        try:
            with open(patch_file, "w") as f:
                f.write(patch if patch.endswith("\n") else patch + "\n")

            # First try with git apply
            git_apply_cmd = f"git apply -v {str(patch_file)}"
            stdout, git_apply_status = await self._execute_command(git_apply_cmd)

            # Check if we have changes by looking at both tracked changes and untracked files
            diff, _ = await self._execute_command("git diff")
            untracked, _ = await self._execute_command("git ls-files --others --exclude-standard")

            # If we have either tracked changes or new untracked files, the patch was applied successfully
            if diff.strip() or untracked.strip():
                logger.info(f"Git apply succeeded.\n\nOutput:\n{stdout}\n\nGit diff:\n{diff}")
                # Verify that modified/added files actually exist on disk
                if not await self._verify_files_exist(patch):
                    logger.error("Verification failed: Some files from the patch do not exist on disk")
                    return False

                return True

            # If git apply didn't work, try with patch command
            logger.info(f"Git apply failed or made no changes. Output:\n{stdout}")
            logger.info("Trying again with patch command...")

            patch_cmd = f"patch --batch --fuzz=5 -p1 -i {str(patch_file)}"
            stdout, patch_status = await self._execute_command(patch_cmd)

            # Check if patch command worked by looking at both tracked changes and untracked files
            if patch_status == 0:
                diff, _ = await self._execute_command("git diff")
                untracked, _ = await self._execute_command("git ls-files --others --exclude-standard")
                if diff.strip() or untracked.strip():
                    logger.info(f"Patch command succeeded.\n\nOutput:\n{stdout}\n\nGit diff:\n{diff}")
                    # Verify that modified/added files actually exist on disk
                    if not await self._verify_files_exist(patch):
                        logger.error("Verification failed: Some files from the patch do not exist on disk")
                        return False

                    return True

            # If we get here, both methods failed
            logger.info(f"Both git apply and patch command failed. Output:\n{stdout}")
            return False
        finally:
            # Clean up the temporary patch file
            if patch_file.exists():
                os.unlink(patch_file)

    async def _verify_files_exist(self, patch: str) -> bool:
        """
        Verify that all files mentioned in the patch exist on disk.

        Args:
            patch: The patch string to analyze

        Returns:
            bool: True if all files exist, False otherwise
        """
        # Extract target files from the patch
        files_to_check = set()
        # Look for the +++ lines which specify target files
        for line in patch.splitlines():
            if line.startswith("+++"):
                # Skip /dev/null
                if "/dev/null" in line:
                    continue

                # Extract filename, handling paths with a/ or b/ prefixes
                file_path = line[4:].strip()
                if file_path.startswith("b/"):
                    file_path = file_path[2:]

                files_to_check.add(file_path)

        # Check if all files exist
        for file_path in files_to_check:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                logger.error(f"File from patch does not exist: {file_path}")
                return False

        return True

    async def _reset_repository(self) -> bool:
        """Reset the git repository to its original state."""
        reset_commands = [
            "git clean -fd",
            f"git reset --hard {self.swebench_instance['base_commit']}",
        ]

        reset_commands = "\n".join(reset_commands)

        stdout, return_code = await self._execute_command(reset_commands)

        if return_code != 0:
            logger.error(f"Failed to reset repository: {stdout}")
            return False

        return True

    def _test_script(self, test_files: list[str]) -> str:
        directives = [d for d in test_files if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]

        if self.swebench_instance["repo"] == "django/django":
            directives_transformed = []
            for d in directives:
                d = d[: -len(".py")] if d.endswith(".py") else d
                d = d[len("tests/") :] if d.startswith("tests/") else d
                d = d.replace("/", ".")
                directives_transformed.append(d)
            directives = directives_transformed

        test_cmd = (
            MAP_REPO_VERSION_TO_SPECS.get(self.swebench_instance["repo"], {})
            .get(self.swebench_instance["version"], {})
            .get("test_cmd", "")
        )
        return " ".join([test_cmd, *directives])
