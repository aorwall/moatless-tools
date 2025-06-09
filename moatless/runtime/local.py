import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from swebench.harness.constants import SWEbenchInstance, NON_TEST_EXTS
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec, MAP_REPO_VERSION_TO_SPECS

from moatless.environment.base import BaseEnvironment, EnvironmentExecutionError
from moatless.exceptions import RuntimeError
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.storage.base import BaseStorage
from moatless.testing.python.parser_registry import parse_log
from moatless.testing.schema import TestResult
from moatless.context_data import current_node_id
from unidiff import PatchSet

logger = logging.getLogger(__name__)


class SweBenchLocalEnvironment(RuntimeEnvironment, BaseEnvironment):
    """Environment implementation for running swebench tests and evaluations."""

    def __init__(
        self,
        repo_path: Path,
        swebench_instance: SWEbenchInstance,
        storage: BaseStorage,
    ):
        if not repo_path.exists():
            raise RuntimeError(f"Repository path does not exist: {repo_path}")

        if not swebench_instance:
            raise RuntimeError("SWEbench instance is required")

        logger.info(f"Creating LocalEnvironment instance. ID: {swebench_instance['instance_id']}")
        self.repo_path = repo_path
        self.swebench_instance = swebench_instance
        self.instance_id = swebench_instance["instance_id"]
        self.storage = storage
        self.test_spec = make_test_spec(self.swebench_instance)
        self._install_task = None  # Will hold the installation process task
        self._skip_conda_activate = os.getenv("SKIP_CONDA_ACTIVATE", "false").lower() == "true"
        logger.info(f"SKIP_CONDA_ACTIVATE: {self._skip_conda_activate}")

        specs = MAP_REPO_VERSION_TO_SPECS.get(self.swebench_instance["repo"], {}).get(
            self.swebench_instance["version"], {}
        )

        self._install_command = specs.get("install")

        if self.swebench_instance["repo"] == "sphinx-doc/sphinx":
            self._install_after_patch = True
        else:
            self._install_after_patch = False

        self._install_task = asyncio.create_task(self._run_async_installation(self._install_command))

    async def execute(self, command: str, fail_on_error: bool = False, patch: str | None = None) -> str:
        """Execute a command in the environment."""
        await self._wait_for_install()
        try:
            
            if patch:
                await self._apply_patch(patch)
                    
                if self._install_after_patch and self._install_command:
                    logger.info(f"Installing after patch with command: {self._install_command}")
                    stdout, return_code = await self._execute_command(self._install_command)
                    logger.info(f"Installation output: {stdout} {return_code}")
            
            stdout, return_code = await self._execute_command(command)
            logger.info(f"Command {command} returned {return_code}")
            logger.info(f"Output: {stdout}")

            # Save execution log
            if self.storage:
                await self._save_execution_log(command, patch, stdout, return_code)

            if fail_on_error and return_code != 0:
                raise EnvironmentExecutionError(message=f"Command {command} failed with return code {return_code}: {stdout}", stderr=stdout, return_code=return_code)

        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            raise e
        finally:
            if patch:
                await self._reset_repository()

        return stdout

    async def read_file(self, path: str) -> str:
        """Read a file from the environment."""
        stdout, return_code = await self._execute_command(f"cat {path}")
        return stdout

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the environment."""
        stdout, return_code = await self._execute_command(f"echo '{content}' > {path}")
        if return_code != 0:
            raise EnvironmentExecutionError(
                f"Command {f'echo {content} > {path}'} failed with return code {return_code}", return_code, stdout
            )

    async def run_tests(
        self, patch: str | None = None, test_files: list[str] | None = None, timeout: int = 600
    ) -> list[TestResult]:
        """Run tests with an optional patch and specific test files."""
        # Wait for installation to complete if it's running
        await self._wait_for_install()

        logger.info(f"Starting test {test_files}")

        # Apply patch if provided
        if patch:
            if not await self._apply_patch(patch):
                raise RuntimeError("Failed to apply patch")

        if self._install_after_patch and self._install_command:
            logger.info(f"Installing after patch with command: {self._install_command}")
            stdout, return_code = await self._execute_command(self._install_command)
            logger.info(f"Installation output: {stdout} {return_code}")

        test_results = []
        if test_files:  # Check if test_files is not None
            test_spec = self.test_spec

            # Run test files one by one to be sure to know which file the test failed on
            for test_file in test_files:
                test_command = self._test_script([test_file])
                logger.info(f"Test command: {test_command}")

                stdout, return_code = await self._execute_command(test_command, timeout=timeout)

                timed_out = return_code == -1  # -1 indicates timeout

                if timed_out:
                    logger.warning(f"Test timed out after {timeout} seconds for file: {test_file}")

                logger.info(f"Return code: {return_code}")
                if return_code != 0:
                    logger.warning(f"Test output: {stdout}")
                else:
                    logger.debug(f"Test output: {stdout}")

                if self.storage:
                    await self._save_execution_log(test_command, patch, stdout, return_code)

                testbed_results = parse_log(stdout, test_spec.repo)

                # Set timeout flag and ensure file_path is set for all results
                for result in testbed_results:
                    if not result.file_path:
                        result.file_path = test_file
                    if timed_out:
                        result.timed_out = True
                        # If we don't have any results but we timed out, create a basic result
                        if result.status == TestStatus.UNKNOWN and not result.failure_output:
                            result.failure_output = f"Test execution timed out after {timeout} seconds"

                # If no results were parsed but we ran a test (even if it timed out), create a basic result
                if not testbed_results:
                    from moatless.testing.schema import TestResult, TestStatus

                    basic_result = TestResult(
                        status=TestStatus.ERROR if timed_out else TestStatus.UNKNOWN,
                        file_path=test_file,
                        timed_out=timed_out,
                        failure_output=f"Test execution timed out after {timeout} seconds"
                        if timed_out
                        else "No test results could be parsed from output",
                    )
                    testbed_results = [basic_result]

                test_results.extend(testbed_results)

        if patch:
            await self.reset_modified_files(patch)

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
        eval_script = self.make_eval_script()
        eval_file.write_text(eval_script)
        await self.storage.write_raw(f"{evaluation_key}/eval.sh", eval_script)

        logger.debug(f"Executing command: {eval_file} in {self.repo_path}")
        stdout, return_code = await self._execute_command(f"/bin/bash {eval_file}")
        logger.info(f"Eval output: {stdout} {return_code}")

        test_output_path = "/tmp/test_output.log"
        with open(test_output_path, "w") as f:
            f.write(stdout)

        await self.storage.write_raw(f"{evaluation_key}/test_output.txt", stdout)

        report = get_eval_report(
            test_spec=self.test_spec,
            prediction={"instance_id": self.instance_id, "model_patch": patch},
            test_log_path=test_output_path,
            include_tests_status=True,
        )

        await self.storage.write(f"{evaluation_key}/report.json", report)

        if patch:
            await self.reset_modified_files(patch)

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

        stdout, return_code = await self._execute_command("which python")
        logger.info(f"Which python: {stdout} {return_code}")

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

    async def _execute_command(
        self, command: str, cwd: Path | None = None, timeout: int | None = None
    ) -> Tuple[str, int]:
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

        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return stdout.decode(), process.returncode or 0
        except asyncio.TimeoutError:
            # Kill the process if it times out
            process.kill()
            try:
                await process.wait()
            except:
                pass
            # Return partial output if any was captured before timeout
            partial_output = ""
            if process.stdout:
                try:
                    partial_data = await asyncio.wait_for(process.stdout.read(), timeout=1.0)
                    partial_output = partial_data.decode()
                except:
                    pass
            return partial_output, -1  # Use -1 to indicate timeout

    async def _apply_patch(self, patch: str) -> bool:
        """Apply a git patch to the repository."""
        logger.debug("Applying patch to local repository")

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

    async def reset_modified_files(self, patch: str):
        modified_files = self.get_modified_files(patch)
        logger.info(f"Resetting modified files: {modified_files}")

        # Reset only the modified files to their base state
        for modified_file in modified_files:
            reset_command = f"git checkout {self.swebench_instance['base_commit']} {modified_file}"
            stdout, return_code = await self._execute_command(reset_command)
            logger.info(f"Command: {reset_command} Reset output: {stdout} {return_code}")
            if return_code != 0:
                logger.error(f"Failed to reset modified files: {stdout}")

                # Try to clean untracked files and directories if checkout failed
                dir_path = str(Path(modified_file).parent)
                if dir_path:
                    clean_command = f"git clean -fd {dir_path}"
                    stdout, return_code = await self._execute_command(clean_command)
                    logger.info(f"Command: {clean_command} Clean output: {stdout} {return_code}")
                    if return_code != 0:
                        logger.error(f"Failed to clean repository: {stdout}")
                        return False

        return True

    def get_modified_files(self, patch: str) -> list[str]:
        """
        Get the list of modified files in a patch
        """
        source_files = []
        for file in PatchSet(patch):
            if file.source_file != "/dev/null":
                source_files.append(file.source_file)
        source_files = [x[2:] for x in source_files if x.startswith("a/")]
        return source_files

    def make_eval_script(self):
        eval_script_list = self.make_eval_script_list_py()
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"

    def make_eval_script_list_py(self) -> list:
        """
        Applies the test patch and runs the tests.
        """
        from swebench.harness.utils import get_modified_files
        from swebench.harness.test_spec.python import get_test_directives
        from swebench.harness.constants import (
            MAP_REPO_VERSION_TO_SPECS,
            START_TEST_OUTPUT,
            END_TEST_OUTPUT,
        )

        test_patch = self.swebench_instance["test_patch"]
        base_commit = self.swebench_instance["base_commit"]
        repo_directory = "/testbed"

        HEREDOC_DELIMITER = "EOF_114329324912"
        test_files = get_modified_files(test_patch)
        # Reset test files to the state they should be in before the patch.
        reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
        apply_test_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        test_command = " ".join(
            [
                MAP_REPO_VERSION_TO_SPECS[self.swebench_instance["repo"]][self.swebench_instance["version"]][
                    "test_cmd"
                ],
                *get_test_directives(self.swebench_instance),
            ]
        )
        specs = MAP_REPO_VERSION_TO_SPECS[self.swebench_instance["repo"]][self.swebench_instance["version"]]
        eval_commands = []
        if "eval_commands" in specs:
            eval_commands += specs["eval_commands"]
        eval_commands += [
            f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
            f"cd {repo_directory}",
            f"which python",
            f"python --version",
            # This is just informational, so we have a record
            "git status",
            "git show",
            f"git -c core.fileMode=false diff {base_commit}",
        ]
        # if "install" in specs:
        #    eval_commands.append(specs["install"])
        eval_commands += [
            reset_tests_command,
            apply_test_patch_command]
        
        if self._install_after_patch and self._install_command:
            eval_commands += [self._install_command]
        
        eval_commands += [
            f": '{START_TEST_OUTPUT}'",
            test_command,
            f": '{END_TEST_OUTPUT}'",
            reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
        ]
        return eval_commands

    async def _save_execution_log(self, command: str, patch: str | None, output: str, return_code: int) -> None:
        """Save execution log for the command."""
        try:
            trajectory_path = self.storage.get_trajectory_path()
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Create a smart filename with command snippet and return code
            command_snippet = self._create_command_snippet(command)
            filename = f"{datetime_str}_{command_snippet}_rc{return_code}.log"
            
            log_dir = f"{trajectory_path}/logs"
            
            node_id = current_node_id.get()
            if node_id:
                log_dir = f"{log_dir}/node_{node_id}"
                
            log_path = f"{log_dir}/{filename}"
            
            # Create log content
            log_content = f"Command: {command}\n"
            log_content += f"Return Code: {return_code}\n"
            if patch:
                log_content += f"Patch Applied: Yes\n"
                log_content += f"Patch Content:\n{patch}\n"
                log_content += "=" * 80 + "\n"
            else:
                log_content += "Patch Applied: No\n"
            log_content += f"Output:\n{output}\n"
            
            await self.storage.write_raw(log_path, log_content)
            logger.info(f"Execution log saved to {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")

    def _create_command_snippet(self, command: str) -> str:
        """Create a smart snippet of the command for the filename."""
        # Remove conda activation commands and bash wrapper
        clean_command = command
        if ". /opt/miniconda3/bin/activate && conda activate testbed && " in command:
            clean_command = command.replace(". /opt/miniconda3/bin/activate && conda activate testbed && ", "")
        
        # Take first meaningful part of command
        parts = clean_command.split()
        if not parts:
            return "empty"
        
        # Use first command word and some arguments, but keep it short
        snippet_parts = []
        char_count = 0
        max_chars = 30  # Keep filename reasonable
        
        for part in parts[:4]:  # Take at most 4 parts
            if char_count + len(part) > max_chars:
                break
            snippet_parts.append(part)
            char_count += len(part) + 1  # +1 for space
        
        snippet = "_".join(snippet_parts)
        
        # Clean up for filename safety
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
        snippet = "".join(c if c in safe_chars else "_" for c in snippet)
        
        # Remove consecutive underscores
        while "__" in snippet:
            snippet = snippet.replace("__", "_")
        
        # Trim underscores from start/end
        snippet = snippet.strip("_")
        
        return snippet[:30]  # Ensure max length
