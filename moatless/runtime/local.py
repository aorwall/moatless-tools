import asyncio
import json
import logging
import os
import random
import string
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

from moatless.repository.git import GitRepository
from moatless.storage.base import BaseStorage
from moatless.testing.python.parser_registry import parse_log
from testbeds.schema import (
    EvaluationResult,
    ResolvedStatus,
)
from datetime import datetime
from swebench.harness.grading import get_eval_report

from swebench.harness.test_spec.test_spec import (
    TestSpec as SwebenchTestSpec,
    make_test_spec,
)
from testbeds.swebench.test_spec import TestSpec
from testbeds.swebench.utils import load_swebench_instance
from testbeds.schema import SWEbenchInstance

import swebench
from moatless.exceptions import RuntimeError
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.testing.schema import TestResult

logger = logging.getLogger(__name__)


class LocalEnvironment(RuntimeEnvironment):
    """Local environment implementation for running tests without remote testbed clients."""

    def __init__(
        self,
        repo_path: Path,
        swebench_instance: SWEbenchInstance | None = None,
        instance_id: str | None = None,
        storage: BaseStorage | None = None,
        enable_cache: bool = False,
        run_id: str = None,
        rerun_failing_tests: bool = False,
        include_relevant_files: bool = False,
    ):
        logger.info(f"Creating LocalEnvironment instance. ID: {instance_id}, Run ID: {run_id}")
        self.repo_path = repo_path
        self.swebench_instance = swebench_instance
        self.instance_id = instance_id
        self.storage = storage
        self.run_id = run_id or "".join(random.choices(string.ascii_lowercase, k=6))
        self.rerun_failing_tests = rerun_failing_tests
        self.include_relevant_files = include_relevant_files
        self.tests_to_ignore = []
        self._test_cache = {} if enable_cache else None
        self.test_spec = None
        self._install_task = None  # Will hold the installation process task

        # Kick off the install process in the background
        asyncio.create_task(self._init_install())

    async def _init_install(self):
        """Initialize the environment by running the install command in the background"""
        test_spec = await self._get_test_spec()
        if test_spec and "install" in test_spec.specs:
            self._install_task = asyncio.create_task(self._run_install_command())

    async def _run_install_command(self) -> bool:
        """Run the install command from the test spec."""
        logger.info("Running installation command from test spec")
        test_spec = await self._get_test_spec()
        if not test_spec:
            logger.error("Test spec not available for installation")
            return False

        if "install" not in test_spec.specs:
            logger.info("No install command specified in the test spec")
            return True

        install_command = test_spec.specs["install"]
        logger.info(f"Running install command: {install_command}")

        output, return_code = await self._execute_command(install_command)

        if return_code != 0:
            logger.error(f"Installation failed. Commands: {install_command}\n\nOutput: {output}")
            raise RuntimeError("Installation failed")

        logger.info("Installation completed successfully")
        return True

    async def _wait_for_install(self):
        """Wait for the installation process to complete if it's running."""
        if self._install_task is not None and not self._install_task.done():
            logger.info("Waiting for installation process to complete...")
            try:
                await self._install_task
            except Exception as e:
                logger.error(f"Installation process failed: {str(e)}")
                raise RuntimeError("Installation process failed") from e

    async def _get_test_spec(self) -> Optional[TestSpec]:
        """Retrieve the test specification for the instance."""
        if not self.test_spec:
            instance = None
            if self.swebench_instance:
                instance = self.swebench_instance
            else:
                try:
                    instance = await load_swebench_instance(self.instance_id)
                except Exception as e:
                    logger.error(f"Error loading test spec: {str(e)}")
                    return None

            if instance:
                self.test_spec = TestSpec.from_instance(instance)

        return self.test_spec

    async def get_swenbench_test_spec(self) -> Optional[SwebenchTestSpec]:
        """Retrieve the test specification for the instance."""
        try:
            instance = await load_swebench_instance(self.instance_id)
            if instance:
                return make_test_spec(instance.model_dump())
        except Exception as e:
            logger.error(f"Error loading test spec: {str(e)}")
            return None

    async def _execute_command(self, command: str, cwd: Path = None) -> Tuple[str, int]:
        """Execute a shell command and return combined output and return code."""

        # Prepend conda activation to the command using . instead of source
        conda_activate = ". /opt/miniconda3/bin/activate && conda activate testbed && "
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

    async def _execute_command_test(self, command: str) -> Tuple[str, int]:
        """Execute a shell command and return combined output and return code."""

        # Create a temporary script file to ensure proper environment activation and output capture
        script_path = Path("/tmp/moatless_command_script.sh")

        # Prepare the script content with proper environment setup
        script_content = f"""#!/bin/bash
set -e

# Source conda setup and activate environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate testbed

# Print debug info
echo "--- Debug Info Start ---"
echo "Current PATH: $PATH"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
which conda
conda info
which python
python --version
which pip
pip --version
echo "--- Debug Info End ---"

# Install package

echo "Current directory: $(pwd)"

# Install package
echo "=== Starting package installation ==="
echo "Executing command: {command}"

{command}
"""

        # Write script to file
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make it executable
        os.chmod(script_path, 0o755)

        logger.info(f"Executing command via script: {command}")

        # Note: Not passing os.environ.copy() to avoid environment conflicts
        # Use login shell (-l) for proper environment initialization
        process = await asyncio.create_subprocess_shell(
            f"bash -l {script_path}",
            cwd="/testbed",  # Use /testbed as the working directory
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # redirect stderr to stdout
        )

        stdout, _ = await process.communicate()

        # Clean up the temporary script
        if script_path.exists():
            os.unlink(script_path)

        return stdout.decode(), process.returncode or 0

    async def _apply_patch(self, patch: str) -> bool:
        """Apply a git patch to the repository."""
        logger.info("Applying patch to local repository")
        await self._reset_repository()

        test_spec = await self._get_test_spec()
        if not test_spec:
            raise RuntimeError("Test spec not available")

        # Save patch to a temporary file
        patch_file = self.repo_path / "temp_patch.diff"
        try:
            with open(patch_file, "w") as f:
                f.write(patch if patch.endswith("\n") else patch + "\n")

            # Execute patch commands using the commands from test_spec
            patch_commands = test_spec.patch_commands(str(patch_file))
            if isinstance(patch_commands, list):
                patch_commands = "\n".join(patch_commands)

            stdout, return_code = await self._execute_command(patch_commands)

            if return_code != 0:
                diff = await self._execute_command("git diff")
                logger.error(
                    f"Failed to apply patch.\n\nCommands: {patch_commands}\n\nOutput: {stdout}.\n\nGit diff: {diff}"
                )
                return False

            return True
        finally:
            # Clean up the temporary patch file
            if patch_file.exists():
                os.unlink(patch_file)

    async def _reset_repository(self) -> bool:
        """Reset the git repository to its original state."""
        test_spec = await self._get_test_spec()
        if not test_spec:
            raise RuntimeError("Test spec not available")

        reset_commands = test_spec.reset_commands
        if isinstance(reset_commands, list):
            reset_commands = "\n".join(reset_commands)

        stdout, return_code = await self._execute_command(reset_commands)

        if return_code != 0:
            logger.error(f"Failed to reset repository: {stdout}")
            return False

        return True

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

        test_results = []
        if test_files:  # Check if test_files is not None
            test_spec = await self._get_test_spec()
            if not test_spec:
                raise RuntimeError("Test spec not available")

            for test_file in test_files:
                test_commands = test_spec.test_script([test_file])
                logger.info(f"Test command: {test_commands}")

                stdout, return_code = await self._execute_command(test_commands[1])

                logger.info(f"Test output: {stdout}")

                if self.storage:
                    trajectory_key = self.storage.get_trajectory_key()
                    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log_path = f"{trajectory_key}/logs/{datetime_str}_test_run.log"
                    await self.storage.write_raw(log_path, stdout)
                    logger.info(f"Test log saved to {log_path}")

                testbed_results = parse_log(stdout, test_spec.repo)
                test_results.extend(testbed_results)

        return test_results

    async def evaluate(self, patch: str) -> Union[EvaluationResult, None]:
        """
        Run evaluation with the provided patch.

        This method applies the patch and runs the evaluation script to check if
        the patch resolves the issue as expected.

        Args:
            patch: The patch to evaluate

        Returns:
            EvaluationResult or None if evaluation fails
        """
        # Wait for installation to complete if it's running
        await self._wait_for_install()

        logger.info(f"Starting evaluation for instance {self.instance_id} with run_id {self.run_id}")

        log_content = ""
        try:
            if not patch.endswith("\n"):
                patch += "\n"

            log_content += f"\n\n# Patch:\n```diff\n{patch}\n```"

            # Reset the repository
            if not await self._reset_repository():
                logger.error("Failed to reset repository for evaluation")
                return None

            # Apply the patch
            if not await self._apply_patch(patch):
                logger.error("Failed to apply patch for evaluation")
                return None

            # Get the test spec
            test_spec = await self._get_test_spec()
            if not test_spec:
                logger.error("Test spec not available for evaluation")
                return None

            # Run evaluation command
            eval_commands = test_spec.eval_script_list
            if isinstance(eval_commands, list):
                eval_commands = "\n".join(eval_commands)

            logger.info(f"Eval commands: {eval_commands}")
            log_content += f"\n\n# Eval commands:\n```\n{eval_commands}\n```"

            stdout, return_code = await self._execute_command(eval_commands)
            logger.info(f"Eval output: {stdout} {return_code}")
            log_content += f"\n\n# Eval output:\n```\n{stdout}\n```"

            test_status = test_spec.get_pred_report(stdout)

            return EvaluationResult(
                run_id="",
                resolved=test_status.status == ResolvedStatus.FULL,
                patch_applied=True,
                instance_id=self.instance_id,
                tests_status=test_status,
            )
        except Exception as e:
            logger.exception("Error running evaluation")
            log_content += f"\n\n## Error:\n{e}"
            import traceback

            traceback_text = traceback.format_exc()
            log_content += f"\n\n# Traceback:\n{traceback_text}"
        finally:
            if self.storage:
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                trajectory_key = self.storage.get_trajectory_key()
                log_path = f"{trajectory_key}/logs/{datetime_str}_evaluation.md"
                await self.storage.write_raw(log_path, log_content)
                logger.info(f"Evaluation log saved to {log_path}")

        return None

    async def swebench_evaluate(self, patch: str):
        """
        Run evaluation with the provided patch.

        This method applies the patch and runs the evaluation script to check if
        the patch resolves the issue as expected.

        Args:
            patch: The patch to evaluate

        Returns:
            EvaluationResult or None if evaluation fails
        """
        # Wait for installation to complete if it's running
        await self._wait_for_install()

        from testbeds.swebench.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )

        logger.info(f"Starting evaluation for instance {self.instance_id} with run_id {self.run_id}")

        log_content = ""
        try:
            if not patch.endswith("\n"):
                patch += "\n"

            log_content += f"\n\n# Patch:\n```diff\n{patch}\n```"

            # Reset the repository
            if not await self._reset_repository():
                logger.error("Failed to reset repository for evaluation")
                return None

            # Apply the patch
            if not await self._apply_patch(patch):
                logger.error("Failed to apply patch for evaluation")
                return None

            # Get the test spec
            test_spec = await self.get_swenbench_test_spec()
            if not test_spec:
                logger.error("Test spec not available for evaluation")
                return None

            # Run evaluation command
            eval_commands = test_spec.eval_script_list
            if isinstance(eval_commands, list):
                eval_commands = "\n".join(eval_commands)

            stdout, return_code = await self._execute_command(eval_commands)
            logger.info(f"Eval prep output: {stdout} {return_code}")

            if stdout:
                log_content += f"\n\n## Log:\n```\n{stdout}\n```\n"

            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            trajectory_key = self.storage.get_trajectory_key()
            test_output_path = f"{trajectory_key}/logs/{datetime_str}_test_run.log"
            await self.storage.write_raw(test_output_path, stdout)
            logger.info(f"Test log saved to {test_output_path}")

            report = get_eval_report(
                test_spec=test_spec,
                prediction={"instance_id": self.instance_id, "model_patch": patch},
                test_log_path=test_output_path,
                include_tests_status=True,
            )

            logger.info(json.dumps(report, indent=4))

            return report
        except Exception as e:
            logger.exception("Error running evaluation")
            log_content += f"\n\n## Error:\n{e}"
            import traceback

            traceback_text = traceback.format_exc()
            log_content += f"\n\n# Traceback:\n{traceback_text}"
        finally:
            if self.log_dir:
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_path = Path(self.log_dir) / f"{datetime_str}_evaluation.md"
                with open(log_path, "w") as f:
                    f.write(log_content)
                logger.info(f"Evaluation log saved to {log_path}")

        return None

    def clear_cache(self):
        """Clear the test results cache"""
        if self._test_cache is not None:
            self._test_cache.clear()

    def __del__(self):
        """Cleanup when environment is deleted"""
        self.clear_cache()
