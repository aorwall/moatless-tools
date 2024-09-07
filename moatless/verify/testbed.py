import hashlib
import json
import logging
import re
from typing import List

from moatless.file_context import RankedFileSpan
from moatless.verify.verify import Verifier
from testbed.client.client import TestbedClient
from testbed.schema import TestStatus, TestResult, TraceItem

from moatless.repository import GitRepository, CodeFile
from moatless.schema import VerificationIssue, VerificationIssueType, FileWithSpans

logger = logging.getLogger(__name__)


class TestbedVerifier(Verifier):

    def __init__(self, testbed: TestbedClient, repository: GitRepository, max_context_tokens: int = 2000):
        self.testbed = testbed
        self.repository = repository
        self.max_context_tokens = max_context_tokens

    def verify(self, test_files: list[str]) -> List[VerificationIssue]:
        patch = self.repository.diff()
        self.testbed.reset()
        test_results, test_output = self.testbed.run_tests(test_files, patch=patch)

        return self._map_test_results_to_issues(test_results)

    def _get_code_block(self, file_path: str, line_number: int):
        file = self.repository.get_file(file_path)
        if not file or not file.module:
            return None

        block = file.module.find_first_by_start_line(line_number)
        if not block or not block.belongs_to_span:
            return None

        return block

    def _relevant_files_from_trace(self, trace_items: List[TraceItem]) -> List[RankedFileSpan]:
        ranked_file_spans = []

        for i, trace_item in enumerate(trace_items):
            block = self._get_code_block(trace_item.file_path, trace_item.line_number)

            if not block:
                continue

            ranked_file_spans.append(RankedFileSpan(
                file_path=trace_item.file_path,
                span_id=block.belongs_to_span.span_id,
                rank=i,
            ))

        return ranked_file_spans

    def _hash_output(self, output: str):
        """
        Hash only lines with > or E and the last line if it matches the format path:line_number: <Error>
        """
        lines = output.split("\n")
        
        # Regular expression to match the format path:line_number: <Error>
        error_regex = re.compile(r'.+:\d+:.+')
        
        # Check if the last line matches the regex
        if error_regex.match(lines[-1]):
            return hashlib.sha256(lines[-1].encode()).hexdigest()
        
        filtered_out_lines = [line for line in lines if line.startswith("E ") or line.startswith("> ")]
        return hashlib.sha256("\n".join(filtered_out_lines).encode()).hexdigest()

    def _map_test_results_to_issues(self, test_results: List[TestResult]) -> List[VerificationIssue]:
        failures = [result for result in test_results if result.status in [TestStatus.FAILED, TestStatus.ERROR]]
        logger.info(f"{len(failures)} out of {len(test_results)} tests failed.")

        root_causes = set()
        ignored_tests = 0

        issues = []
        for failure in failures:
            trace_items = failure.stacktrace
            # reverse to start from root cause method on ERROR
            if failure.status == TestStatus.ERROR:
                trace_items.reverse()

            if not failure.failure_output:
                logger.warning(f"Skipping test {failure.method} in {failure.file_path} with no failure output")
                continue

            # DeprecationWarnings are probably false negatives because of incorrect dependencies in the testbed environment
            if "DeprecationWarning" in failure.failure_output.split("\n")[-1]:
                logger.info(f"Skipping test {failure.method} in {failure.file_path} with DeprecationWarning")
                continue

            failure_sections = failure.failure_output.split("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
            if len(failure_sections) > 1:
                # skip tests with the same root cause
                hashed_section = self._hash_output(failure_sections[-1])
            elif trace_items and trace_items[0].output:
                hashed_section = hashlib.sha256((str(trace_items[0])).encode()).hexdigest()
            else:
                hashed_section = self._hash_output(failure.failure_output)

            if hashed_section in root_causes:
                ignored_tests += 1
                continue

            relevant_files = self._relevant_files_from_trace(trace_items)

            if not failure.file_path or not failure.method:

                # Use the file with the root cause for the error if no file path or method is set
                if failure.status == TestStatus.ERROR and relevant_files:
                    failure.file_path = relevant_files[0].file_path
                    failure.method = relevant_files[0].span_id
                    logger.info(f"No filepath found on test \"{failure.name}\". Using file path {failure.file_path} and {failure.method} from trace for test {failure.name}")
                else:
                    logger.warning(f"Could not find file path and/or method for test {failure}")
                    continue

            method = failure.method
            if "[" in method:
                method = method.split("[")[0]

            file = self.repository.get_file(failure.file_path)
            if not file and trace_items:
                for item in trace_items:
                    file = self.repository.get_file(item.file_path)
                    if file:
                        method = item.method
                        break

            if not file:
                logger.warning(f"Could not find file {failure.file_path} in test \"{failure.name}\"")
            elif not file.module:
                logger.warning(f"Could not parse file {failure.file_path} in test \"{failure.name}\"")
            else:

                block = None
                if "." in method:
                    path = method.split(".")
                    logger.debug(f"Looking for path {path} in file {file.file_path}")
                    block = file.module.find_by_path(path)
                    if not block:
                        method = path[-1]

                if method == "<module>" and trace_items:
                    block = file.module.children[0]
                    for item in trace_items:
                        if item.file_path == file.file_path and item.line_number:
                            block = file.module.find_first_by_start_line(item.line_number)
                            break

                if not block:
                    block = file.module.find_by_identifier(method, recursive=True)

                if block:
                    span_id = block.belongs_to_span.span_id
                    existing_issue = next((issue for issue in issues if issue.span_id == span_id and issue.file_path == file.file_path), None)
                    if existing_issue:
                        logger.debug(f"Skipping duplicate span id {span_id} for failure in {file.file_path} and method {method}.")
                        continue
                else:
                    span_id = None

                output_length = len(failure.failure_output) if failure.failure_output else 0
                logger.debug(f"Add verification issue with failure in {file.file_path} and {span_id}. Output length: {output_length}")
                issues.append(VerificationIssue(
                    type=VerificationIssueType.TEST_FAILURE if failure.status == TestStatus.FAILED else VerificationIssueType.RUNTIME_ERROR,
                    message=failure.failure_output if failure.failure_output else f"Test {failure.name} failed",
                    file_path=file.file_path,
                    span_id=span_id,
                    relevant_files=relevant_files,
                ))
                root_causes.add(hashed_section)

        if ignored_tests:
            logger.info(f"Ignored {ignored_tests} tests with redundant root cause")
        return issues
