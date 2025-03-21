import hashlib
import json
import logging
import random
import re
import string
from datetime import datetime
from typing import List

from opentelemetry import trace
from testbeds.schema import EvaluationResult, TraceItem, TestbedSummary
from testbeds.sdk import TestbedSDK
from testbeds.sdk.exceptions import TestbedError

from moatless.exceptions import RuntimeError
from moatless.repository import GitRepository
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.testing.schema import TestResult, TestStatus
from moatless.schema import RankedFileSpan

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class TestbedEnvironment(RuntimeEnvironment):
    def __init__(
        self,
        repository: Repository,
        testbed_sdk: TestbedSDK | None = None,
        instance: dict | None = None,
        instance_id: str | None = None,
        log_dir: str | None = None,
        enable_cache: bool = False,
        run_id: str | None = None,
        rerun_failing_tests: bool = False,
        include_relevant_files: bool = False,
    ):
        # Add diagnostic logging
        logger.info(f"Creating TestbedEnvironment instance. ID: {instance_id}, Run ID: {run_id}")

        self.testbed_sdk = testbed_sdk or TestbedSDK(enable_cache=enable_cache)
        self.repository = repository
        self.instance = instance
        if instance_id:
            self.instance_id = instance_id
        else:
            self.instance_id = instance["instance_id"] if instance else None

        if not self.instance_id:
            raise RuntimeError("No instance ID or testbed ID provided")

        self.tests_to_ignore = []
        self.log_dir = log_dir
        self._test_cache = {} if enable_cache else None
        self.run_id = run_id or "".join(random.choices(string.ascii_lowercase, k=6))
        self._client = None
        self.rerun_failing_tests = rerun_failing_tests
        self.include_relevant_files = include_relevant_files
        self.testbed_id = None

    @classmethod
    def from_instance(cls, instance: dict, repository: GitRepository, **kwargs):
        return cls(testbed_sdk=TestbedSDK(), repository=repository, instance=instance, **kwargs)

    @tracer.start_as_current_span("TestbedEnvironment.__aenter__")
    async def __aenter__(self):
        """Enter the async context manager."""
        logger.info(f"Setting up testbed environment for instance {self.instance_id}")

        if self.testbed_id:
            logger.info(f"Using existing testbed ID: {self.testbed_id}")
        else:
            client = await self.testbed_sdk.create_async_client(
                instance_id=self.instance_id, log_dir=self.log_dir, run_id=self.run_id
            )

            try:
                testbed = await client.get_or_create_testbed_async()
                self.testbed_id = testbed.testbed_id
                logger.info(f"Created new testbed with ID: {self.testbed_id}")

            finally:
                # Don't destroy the client here since we're just getting the testbed ID
                # and not performing other operations
                pass

        return self

    @tracer.start_as_current_span("TestbedEnvironment.__aexit__")
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager."""
        if self.testbed_id:
            logger.info(f"Destroying testbed with ID: {self.testbed_id}")
            client = await self.testbed_sdk.create_async_client(
                instance_id=self.instance_id, testbed_id=self.testbed_id, log_dir=self.log_dir, run_id=self.run_id
            )
            try:
                await client.destroy()
                logger.info(f"Testbed {self.testbed_id} destroyed")
            except Exception as e:
                logger.error(f"Error destroying testbed {self.testbed_id}: {e}")

    @tracer.start_as_current_span("TestbedEnvironment.run_tests")
    async def run_tests(self, patch: str | None = None, test_files: list[str] | None = None) -> list[TestResult]:
        async with await self.testbed_sdk.create_async_client(
            instance_id=self.instance_id,
            testbed_id=self.testbed_id,
            log_dir=self.log_dir,
            run_id=self.run_id,
        ) as client:
            logger.info(f"Starting test run for instance {self.instance_id} with run_id {self.run_id}.")
            response = await client.run_tests(test_files=test_files, patch=patch, timeout=600)
            logger.info(f"Test run response test results: {len(response.test_results)}")
        log_content = ""

        if response.output:
            log_content = "# Test Run\n\n"
            log_content += f"Files: {test_files}"
            if patch:
                log_content += f"\n\n# Patch:\n```diff\n{patch}\n```"
            log_content += f"\n\n## Log:\n{response.output}\n"

        if response.test_results:
            log_content += "\n\n## Testbed test results:"
            test_results_json = response.model_dump_json(exclude={"output"}, indent=2)
            log_content += f"```json\n{test_results_json}\n```"

        # Filter using cached tests first
        test_results = self._filter_failing_tests(response.test_results, patch=patch)

        # Now check for failures only in the filtered results
        if (
            self.rerun_failing_tests
            and patch
            and any(test.status in [TestStatus.ERROR, TestStatus.FAILED] for test in test_results)
        ):
            # Only run baseline tests if we haven't cached any failing tests yet
            if not self.tests_to_ignore:
                # Get list of failing test files
                failing_test_files = {
                    test.file_path for test in test_results if test.status in [TestStatus.ERROR, TestStatus.FAILED]
                }

                async with await self.testbed_sdk.create_async_client(
                    instance_id=self.instance_id,
                    testbed_id=self.testbed_id,
                    log_dir=self.log_dir,
                    run_id=self.run_id,
                ) as client:
                    baseline_response = await client.run_tests(
                        test_files=list(failing_test_files), patch=None, timeout=600
                    )
                    logger.info(f"Baseline response test results: {len(baseline_response.test_results)}")

                self._filter_failing_tests(baseline_response.test_results, patch=None)
                # Re-filter the results with any newly cached tests
                test_results = self._filter_failing_tests(response.test_results, patch=patch)

        mapped_results = await self._map_test_results_to_issues(test_results)

        # Cache results if caching is enabled
        if self._test_cache is not None:
            cache_key = self._generate_cache_key(test_files, patch)
            self._test_cache[cache_key] = mapped_results

        return mapped_results

    @tracer.start_as_current_span("TestbedEnvironment.evaluate")
    async def evaluate(self, patch: str) -> EvaluationResult | None:
        async with await self.testbed_sdk.create_async_client(
            instance_id=self.instance_id,
            testbed_id=self.testbed_id,
            log_dir=self.log_dir,
            run_id=self.run_id,
        ) as client:
            logger.info(f"Starting evaluation for instance {self.instance_id} with run_id {self.run_id}.")

            log_content = ""
            try:
                if not patch.endswith("\n"):
                    patch += "\n"

                log_content += f"\n\n# Patch:\n```diff\n{patch}\n```"

                evaluation_result = await client.run_evaluation(patch=patch)

                if evaluation_result.output:
                    log_content += f"\n\n## Log:\n```\n{evaluation_result.output}\n```\n"

                log_content += f"\n\n## Evaluation result:\n```json\n{evaluation_result.model_dump_json(indent=2)}\n```"

                return evaluation_result

            except TestbedError as e:
                logger.error(f"Error running evaluation. Cause: {e}")
                log_content += f"\n\n## Error:\n{e}"
            except Exception as e:
                logger.exception("Error running evaluation")
                log_content += f"\n\n## Error:\n{e}"
                import traceback

                traceback = traceback.format_exc()
                log_content += f"\n\n# Traceback:\n{traceback}"
            finally:
                if self.log_dir:
                    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    with open(f"{self.log_dir}/{datetime_str}_evaluation.md", "w") as f:
                        f.write(log_content)

        return None

    def _generate_cache_key(self, test_files: list[str] | None, patch: str | None = None) -> str:
        """Generate a unique cache key based on test files and patch content"""
        key_parts = []
        if test_files:
            key_parts.extend(sorted(test_files))
        if patch:
            key_parts.append(patch)
        if not key_parts:
            key_parts.append("all_tests_no_patch")

        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _filter_failing_tests(self, test_results: list[TestResult], patch: str | None = None) -> list[TestResult]:
        """
        Filter out tests that fail without any changes to isolate patch-specific failures.

        This function serves two purposes:
        1. When no patch is provided (baseline run), it identifies and caches tests that fail
           due to environment issues, dependencies, or pre-existing bugs
        2. When a patch is provided, it filters out these known failing tests to focus on
           failures caused by the patch itself

        Args:
            test_results: List of test results to filter
            patch: Optional patch being tested. If None, indicates a baseline run

        Returns:
            List[TestResult]: Filtered test results, excluding tests that fail in baseline
        """
        # Check if there are any failures or errors
        has_failures = any(test.status in [TestStatus.ERROR, TestStatus.FAILED] for test in test_results)

        # If no patch and failures exist, cache them for future filtering
        if not patch and has_failures:
            self.tests_to_ignore = [
                test.name for test in test_results if test.status in [TestStatus.ERROR, TestStatus.FAILED]
            ]
            if self.tests_to_ignore and self.log_dir:
                with open(f"{self.log_dir}/ignored_tests.json", "w") as f:
                    json.dump(self.tests_to_ignore, f)
                logger.info(
                    f"Baseline run: Found {len(self.tests_to_ignore)} failing tests that will be ignored in future runs"
                )

        # Filter out ignored tests
        filtered_results = [test for test in test_results if test.name not in self.tests_to_ignore]

        if patch and self.tests_to_ignore:
            logger.info(
                f"Using cached baseline failures: Filtered out {len(test_results) - len(filtered_results)} known failing tests"
            )

        return filtered_results

    async def _get_code_block(self, file_path: str, line_number: int):
        file = self.repository.get_file(file_path)
        if not file:
            return None

        module = await file.async_module()
        if not module:
            return None

        block = module.find_first_by_start_line(line_number)
        if not block or not block.belongs_to_span:
            return None

        return block

    async def _relevant_files_from_trace(self, trace_items: list[TraceItem]) -> list[RankedFileSpan]:
        ranked_file_spans = []
        seen_spans = set()

        for i, trace_item in enumerate(trace_items):
            block = await self._get_code_block(trace_item.file_path, trace_item.line_number)

            if not block:
                continue

            span_key = (trace_item.file_path, block.belongs_to_span.span_id)
            if span_key in seen_spans:
                continue

            seen_spans.add(span_key)
            ranked_file_spans.append(
                RankedFileSpan(
                    file_path=trace_item.file_path,
                    span_id=block.belongs_to_span.span_id,
                    rank=i,
                )
            )

        return ranked_file_spans

    def _hash_output(self, output: str):
        """
        Hash only lines with > or E and the last line if it matches the format path:line_number: <Error>
        """
        lines = output.split("\n")

        # Regular expression to match the format path:line_number: <Error>
        error_regex = re.compile(r".+:\d+:.+")

        # Check if the last line matches the regex
        if error_regex.match(lines[-1]):
            return hashlib.sha256(lines[-1].encode()).hexdigest()

        filtered_out_lines = [line for line in lines if line.startswith("E ") or line.startswith("> ")]
        return hashlib.sha256("\n".join(filtered_out_lines).encode()).hexdigest()

    async def _map_test_results_to_issues(self, test_results: list) -> list[TestResult]:
        file_cache = {}

        def get_cached_file(file_path: str):
            if file_path not in file_cache:
                file_cache[file_path] = self.repository.get_file(file_path)
            return file_cache[file_path]

        root_causes = set()
        ignored_tests = 0

        mapped_results = []
        for test_result in test_results:
            trace_items = test_result.stacktrace

            test_status = TestStatus(test_result.status)

            if test_status not in [TestStatus.ERROR, TestStatus.FAILED]:
                mapped_results.append(
                    TestResult(
                        status=test_status,
                        message=test_result.name,
                        file_path=test_result.file_path,
                    )
                )
                continue

            # reverse to start from root cause method on ERROR
            if test_status == TestStatus.ERROR:
                trace_items.reverse()

            hashed_section = None

            ignored_errors = ["PermissionError not raised", "DeprecationWarning"]

            # DeprecationWarnings are probably false negatives because of incorrect dependencies in the testbed environment
            if test_result.failure_output:
                last_line = [
                    line
                    for line in test_result.failure_output.split("\n")
                    if line.strip() and not line.startswith("_____")
                ][-1]
                if any(error in last_line for error in ignored_errors):
                    logger.info(
                        f"Skipping test {test_result.method} in {test_result.file_path} with ignored error on last line: '{last_line}'"
                    )
                    continue

            if not test_result.failure_output:
                logger.info(f"Skipping test {test_result.method} in {test_result.file_path} with no failure output")
                test_output = None
            else:
                failure_sections = test_result.failure_output.split(
                    "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
                )
                if len(failure_sections) > 1:
                    # skip tests with the same root cause
                    hashed_section = self._hash_output(failure_sections[-1])
                elif trace_items and trace_items[0].output:
                    hashed_section = hashlib.sha256((str(trace_items[0])).encode()).hexdigest()
                else:
                    hashed_section = self._hash_output(test_result.failure_output)

                if hashed_section in root_causes:
                    ignored_tests += 1
                    test_output = None
                else:
                    # If the test has more than 50 lines just pick the last 50
                    if len(test_result.failure_output.split("\n")) > 50:
                        test_output = "\n".join(test_result.failure_output.split("\n")[-50:])
                    else:
                        test_output = test_result.failure_output

            relevant_files = []
            if self.include_relevant_files:
                relevant_files = await self._relevant_files_from_trace(trace_items)

            if test_result.method:
                method = test_result.method
                if "[" in method:
                    method = method.split("[")[0]
            else:
                method = None

            file = None
            if test_result.file_path:
                file = get_cached_file(test_result.file_path)

            if not file:
                mapped_results.append(
                    TestResult(
                        status=test_status,
                        message=test_output,
                        file_path=test_result.file_path,
                        relevant_files=relevant_files,
                    )
                )
                continue

            if file and file.module and method:
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
                    existing_issue = next(
                        (
                            issue
                            for issue in mapped_results
                            if issue.span_id == span_id and issue.file_path == file.file_path
                        ),
                        None,
                    )
                    if existing_issue:
                        logger.debug(
                            f"Skipping content on duplicate span id {span_id} for failure in {file.file_path} and method {method}."
                        )
                        test_output = None
                else:
                    span_id = None

                mapped_results.append(
                    TestResult(
                        status=test_status,
                        message=test_output,
                        file_path=file.file_path,
                        span_id=span_id,
                        relevant_files=relevant_files,
                    )
                )

                if hashed_section:
                    root_causes.add(hashed_section)

            elif test_output:
                mapped_results.append(
                    TestResult(
                        status=test_status,
                        message=test_output,
                        file_path=test_result.file_path,
                        relevant_files=relevant_files,
                    )
                )
            elif test_status in [TestStatus.ERROR, TestStatus.FAILED]:
                logger.warning(
                    f'Could not find file {test_result.file_path} or method in test "{test_result.name}" and no output exists, will ignore'
                )
            else:
                logger.info(f"Skipping test {test_result.name} with status {test_status}")

        if ignored_tests:
            logger.info(f"Ignored {ignored_tests} tests with redundant root cause")

        logger.info(f"Finished mapping {len(test_results)} results to {len(mapped_results)} issues")
        return mapped_results

    def clear_cache(self):
        """Clear the test results cache"""
        if self._test_cache is not None:
            self._test_cache.clear()

    def __del__(self):
        """Cleanup when environment is deleted"""
        self.clear_cache()
