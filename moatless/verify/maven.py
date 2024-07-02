import logging
import os
import re
import subprocess
from typing import Optional

from moatless.repository import CodeFile
from moatless.types import VerificationError
from moatless.verify.verify import Verifier

logger = logging.getLogger(__name__)


class MavenVerifier(Verifier):

    def __init__(self, repo_dir: str, run_tests: bool = True):
        self.repo_dir = repo_dir
        self.run_tests = run_tests

    def verify(self, file: Optional[CodeFile] = None) -> list[VerificationError]:
        try:
            # os.environ["JAVA_HOME"] = "/home/albert/.sdkman/candidates/java/17.0.8-tem"

            version = "21-tem"

            sdkman_cmd = (
                f"source $HOME/.sdkman/bin/sdkman-init.sh && sdk use java {version}"
            )

            if self.run_tests:
                mvn_cmd = "./mvnw clean test"
            else:
                mvn_cmd = "./mvnw clean compile test-compile"

            logger.info(
                f"Running Maven command: {mvn_cmd} with Java version {version} in {self.repo_dir}"
            )
            result = subprocess.run(
                f"{sdkman_cmd} && {mvn_cmd}",
                cwd=self.repo_dir,
                check=False,
                text=True,
                shell=True,
                capture_output=True,
            )

            stdout = result.stdout
            stderr = result.stderr

            combined_output = stdout + "\n" + stderr
            compilation_errors = self.parse_compilation_errors(combined_output)
            if compilation_errors or not self.run_tests:
                return compilation_errors

            test_failures = self.parse_test_failures(combined_output)
            return test_failures

        except subprocess.CalledProcessError as e:
            logger.warning("Error running Maven command:")
            logger.warning(e.stderr)

    def parse_compilation_errors(self, output: str) -> list[VerificationError]:
        error_pattern = re.compile(r"\[ERROR\] (.*?):\[(\d+),(\d+)\] (.*)")
        matches = error_pattern.findall(output)

        errors = []
        for match in matches:
            file_path, line, column, message = match

            file_path = file_path.replace(f"{self.repo_dir}/", "")
            error = VerificationError(
                code="COMPILATION_ERROR",
                file_path=file_path.strip(),
                message=message.strip(),
                line=int(line),
            )
            errors.append(error)
        return errors

    def find_file(self, class_name: str) -> str:
        for root, _, files in os.walk(self.repo_dir):
            for file in files:
                if file == f"{class_name}.java":
                    absolute_path = os.path.join(root, file)
                    return os.path.relpath(absolute_path, self.repo_dir)
        return ""

    def parse_test_failures(self, output: str) -> list[VerificationError]:
        failure_pattern = re.compile(r"\[ERROR\]   (.*?):(\d+) (.*)")
        matches = failure_pattern.findall(output)

        errors = []
        for match in matches:
            test_case, line, message = match

            class_name = test_case.split(".")[0]

            file_path = self.find_file(class_name)

            error = VerificationError(
                code="TEST_FAILURE",
                file_path=file_path.strip(),
                message=message.strip(),
                line=int(line),
            )
            errors.append(error)
        return errors
