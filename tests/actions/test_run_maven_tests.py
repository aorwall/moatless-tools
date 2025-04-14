import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from moatless.actions.run_maven_tests import RunMavenTests, RunMavenTestsArgs
from moatless.environment.base import BaseEnvironment
from moatless.file_context import FileContext
from moatless.testing.schema import TestStatus
from moatless.workspace import Workspace


class TestRunMavenTestsAction:
    @pytest.fixture
    def file_context(self):
        mock_file_context = MagicMock(spec=FileContext)
        mock_file_context.file_exists = lambda path: True
        # Set shadow_mode to False to allow tests to run
        mock_file_context.shadow_mode = False
        mock_file_context.add_test_files = MagicMock()
        return mock_file_context

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment that simulates Maven command execution."""
        mock_env = AsyncMock(spec=BaseEnvironment)

        # Define a helper function to generate Maven-like output based on the command
        async def execute_mock(command, fail_on_error=False):
            if "mvn --version" in command:
                return "Apache Maven 3.8.6 (84538c9988a25aec085021c365c560670ad80f63)"

            if "pwd" in command:
                return "/path/to/project"

            if "[ -d" in command:
                path = command.split("[ -d ")[1].split(" ]")[0]
                if path == "src/test/java/com/example":
                    return "true"
                return "false"

            if "mvn compile" in command:
                # Check if we should simulate a compilation failure
                if getattr(execute_mock, "simulate_compilation_failure", False):
                    return self._get_maven_compilation_failure_output()
                # Default to successful compilation
                return "[INFO] BUILD SUCCESS"

            if "mvn test" in command:
                # Extract the test class if specified with -Dtest
                test_class = None
                if "-Dtest=" in command:
                    test_class = command.split("-Dtest=")[1].split()[0]

                # Simulate a successful test
                if test_class and "SuccessTest" in test_class:
                    return self._get_maven_success_output(test_class)
                # Simulate a test with failures
                elif test_class and "FailTest" in test_class:
                    return self._get_maven_failure_output(test_class)
                # Simulate a test with errors
                elif test_class and "ErrorTest" in test_class:
                    return self._get_maven_error_output(test_class)
                # Default case - run all tests
                else:
                    return self._get_maven_mixed_output()

            # Default response for other commands
            return ""

        mock_env.execute = execute_mock

        # Define read_file behavior - all files exist except for NonExistentTest
        async def read_file_mock(file_path):
            if "NonExistentTest" in file_path:
                raise FileNotFoundError(f"File not found: {file_path}")
            return "file content"

        mock_env.read_file = read_file_mock

        return mock_env

    @pytest.fixture
    def workspace(self, mock_environment):
        """Create a workspace with mock environment."""
        workspace = MagicMock(spec=Workspace)
        workspace.environment = mock_environment
        return workspace

    @pytest.fixture
    def run_maven_tests_action(self, workspace):
        """Create a RunMavenTests action with the workspace."""
        action = RunMavenTests(max_output_tokens=2000)
        asyncio.run(action.initialize(workspace))
        return action

    def _get_maven_success_output(self, test_class):
        """Generate Maven output for a successful test."""
        return f"""
[INFO] Scanning for projects...
[INFO] 
[INFO] ------------------------< com.example:my-app >-------------------------
[INFO] Building my-app 1.0-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- maven-surefire-plugin:2.22.1:test (default-test) @ my-app ---
[INFO] 
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running {test_class}
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.185 s - in {test_class}
[INFO] 
[INFO] Results:
[INFO] 
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  3.461 s
[INFO] Finished at: 2023-05-15T14:22:18-07:00
[INFO] ------------------------------------------------------------------------
"""

    def _get_maven_failure_output(self, test_class):
        """Generate Maven output for a test with failures."""
        return f"""
[INFO] Scanning for projects...
[INFO] 
[INFO] ------------------------< com.example:my-app >-------------------------
[INFO] Building my-app 1.0-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- maven-surefire-plugin:2.22.1:test (default-test) @ my-app ---
[INFO] 
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running {test_class}
[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.289 s <<< FAILURE! - in {test_class}
[ERROR] {test_class}#testSomething  Time elapsed: 0.121 s  <<< FAILURE!
java.lang.AssertionError: expected:<true> but was:<false>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.failNotEquals(Assert.java:835)
    at org.junit.Assert.assertEquals(Assert.java:120)
    at org.junit.Assert.assertEquals(Assert.java:146)
    at {test_class}.testSomething({test_class}.java:42)

[INFO] 
[INFO] Results:
[INFO] 
[ERROR] Failures: 
[ERROR]   {test_class}.testSomething:42 expected:<true> but was:<false>
[INFO] 
[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  3.756 s
[INFO] Finished at: 2023-05-15T14:25:32-07:00
[INFO] ------------------------------------------------------------------------
"""

    def _get_maven_error_output(self, test_class):
        """Generate Maven output for a test with errors."""
        return f"""
[INFO] Scanning for projects...
[INFO] 
[INFO] ------------------------< com.example:my-app >-------------------------
[INFO] Building my-app 1.0-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- maven-surefire-plugin:2.22.1:test (default-test) @ my-app ---
[INFO] 
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running {test_class}
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.275 s <<< FAILURE! - in {test_class}
[ERROR] {test_class}#testWithError  Time elapsed: 0.134 s  <<< FAILURE!
java.lang.AssertionError: expected:<true> but was:<false>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.failNotEquals(Assert.java:835)
    at org.junit.Assert.assertEquals(Assert.java:120)
    at org.junit.Assert.assertEquals(Assert.java:146)
    at {test_class}.testWithError({test_class}.java:58)

[INFO] 
[INFO] Results:
[INFO] 
[ERROR] Failures: 
[ERROR]   {test_class}.testWithError:58 expected:<true> but was:<false>
[INFO] 
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  3.631 s
[INFO] Finished at: 2023-05-15T14:28:45-07:00
[INFO] ------------------------------------------------------------------------
"""

    def _get_maven_mixed_output(self):
        """Generate Maven output for running multiple test classes with mixed results."""
        return """
[INFO] Scanning for projects...
[INFO] 
[INFO] ------------------------< com.example:my-app >-------------------------
[INFO] Building my-app 1.0-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- maven-surefire-plugin:2.22.1:test (default-test) @ my-app ---
[INFO] 
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running com.example.SuccessTest
[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.185 s - in com.example.SuccessTest
[INFO] Running com.example.FailTest
[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.289 s <<< FAILURE! - in com.example.FailTest
[ERROR] com.example.FailTest#testSomething  Time elapsed: 0.121 s  <<< FAILURE!
java.lang.AssertionError: expected:<true> but was:<false>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.failNotEquals(Assert.java:835)
    at org.junit.Assert.assertEquals(Assert.java:120)
    at org.junit.Assert.assertEquals(Assert.java:146)
    at com.example.FailTest.testSomething(FailTest.java:42)
[INFO] Running com.example.ErrorTest
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.275 s <<< FAILURE! - in com.example.ErrorTest
[ERROR] com.example.ErrorTest#testWithError  Time elapsed: 0.134 s  <<< FAILURE!
java.lang.AssertionError: expected:<true> but was:<false>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.failNotEquals(Assert.java:835)
    at org.junit.Assert.assertEquals(Assert.java:120)
    at org.junit.Assert.assertEquals(Assert.java:146)
    at com.example.ErrorTest.testWithError(ErrorTest.java:58)

[INFO] 
[INFO] Results:
[INFO] 
[ERROR] Failures: 
[ERROR]   com.example.FailTest.testSomething:42 expected:<true> but was:<false>
[ERROR]   com.example.ErrorTest.testWithError:58 expected:<true> but was:<false>
[INFO] 
[ERROR] Tests run: 6, Failures: 2, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  4.631 s
[INFO] Finished at: 2023-05-15T14:32:45-07:00
[INFO] ------------------------------------------------------------------------
"""

    def _get_maven_compilation_failure_output(self):
        """Generate Maven output for a compilation failure."""
        return """
[INFO] Scanning for projects...
[INFO]
[INFO] --------------------------< com.example:app >--------------------------
[INFO] Building app 0.0.1-SNAPSHOT
[INFO]   from pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- jacoco:0.8.11:prepare-agent (prepare-agent) @ app ---
[INFO] argLine set to -javaagent:/home/user/.m2/repository/org/jacoco/org.jacoco.agent/0.8.11/org.jacoco.agent-0.8.11-runtime.jar=destfile=/home/user/projects/app/target/jacoco.exec
[INFO]
[INFO] --- resources:3.3.1:resources (default-resources) @ app ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] Copying 44 resources from src/main/resources to target/classes
[INFO]
[INFO] --- compiler:3.12.1:compile (default-compile) @ app ---
[INFO] Recompiling the module because of changed source code.
[INFO] Compiling 324 source files with javac [debug parameters release 21] to target/classes
[INFO] /home/user/projects/app/src/main/java/com/example/core/domain/User.java: Some input files use or override a deprecated API.
[INFO] /home/user/projects/app/src/main/java/com/example/core/domain/User.java: Recompile with -Xlint:deprecation for details.
[INFO] /home/user/projects/app/src/main/java/com/example/util/DataLoader.java: Some input files use unchecked or unsafe operations.
[INFO] /home/user/projects/app/src/main/java/com/example/util/DataLoader.java: Recompile with -Xlint:unchecked for details.
[INFO] -------------------------------------------------------------
[WARNING] COMPILATION WARNING :
[INFO] -------------------------------------------------------------
[WARNING] /home/user/projects/app/src/main/java/com/example/domain/Transaction.java:[69,33] @Builder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[WARNING] /home/user/projects/app/src/main/java/com/example/core/domain/BaseEntity.java:[35,23] @SuperBuilder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[WARNING] /home/user/projects/app/src/main/java/com/example/core/domain/Customer.java:[13,1] Generating equals/hashCode implementation but without a call to superclass, even though this class does not extend java.lang.Object. If this is intentional, add '@EqualsAndHashCode(callSuper=false)' to your type.
[INFO] 6 warnings
[INFO] -------------------------------------------------------------
[INFO] -------------------------------------------------------------
[ERROR] COMPILATION ERROR :
[INFO] -------------------------------------------------------------
[ERROR] /home/user/projects/app/src/main/java/com/example/core/api/CustomerController.java:[55,64] cannot infer type arguments for com.example.core.dto.PageResponseDto<>
  reason: cannot infer type-variable(s) T
    (actual and formal argument lists differ in length)
[ERROR] /home/user/projects/app/src/main/java/com/example/core/services/impl/CustomerServiceImpl.java:[143,79] incompatible types: java.util.List<com.example.core.domain.Address> cannot be converted to java.util.Set<com.example.core.domain.Address>
[ERROR] /home/user/projects/app/src/main/java/com/example/core/services/impl/CustomerServiceImpl.java:[148,25] cannot find symbol
  symbol:   method id(java.util.UUID)
  location: class com.example.core.dto.AddressDto.AddressDtoBuilder
[INFO] 3 errors
[INFO] -------------------------------------------------------------
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  2.550 s
[INFO] Finished at: 2025-04-11T13:40:22+02:00
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.12.1:compile (default-compile) on project app: Compilation failure: Compilation failure:
[ERROR] /home/user/projects/app/src/main/java/com/example/core/api/CustomerController.java:[55,64] cannot infer type arguments for com.example.core.dto.PageResponseDto<>
[ERROR]   reason: cannot infer type-variable(s) T
[ERROR]     (actual and formal argument lists differ in length)
[ERROR] /home/user/projects/app/src/main/java/com/example/core/services/impl/CustomerServiceImpl.java:[143,79] incompatible types: java.util.List<com.example.core.domain.Address> cannot be converted to java.util.Set<com.example.core.domain.Address>
[ERROR] /home/user/projects/app/src/main/java/com/example/core/services/impl/CustomerServiceImpl.java:[148,25] cannot find symbol
[ERROR]   symbol:   method id(java.util.UUID)
[ERROR]   location: class com.example.core.dto.AddressDto.AddressDtoBuilder
[ERROR] -> [Help 1]
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR]
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
"""

    @pytest.mark.asyncio
    async def test_run_successful_test(self, run_maven_tests_action, file_context):
        """Test running a successful Maven test."""
        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/SuccessTest.java"],
            thoughts="Testing successful Maven test execution",
        )

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify the response contains the expected summary format
        assert "passed" in result.message
        assert "failed" in result.message
        assert "errors" in result.message
        assert len(result.artifact_changes) == 1
        assert result.artifact_changes[0].properties["passed"] == 2
        assert result.artifact_changes[0].properties["failed"] == 0

    @pytest.mark.asyncio
    async def test_run_failing_test(self, run_maven_tests_action, file_context):
        """Test running a Maven test with failures."""
        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/FailTest.java"], thoughts="Testing Maven test with failures"
        )

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify the response
        assert "AssertionError" in result.message
        assert "expected:<true> but was:<false>" in result.message
        assert len(result.artifact_changes) == 1
        assert result.artifact_changes[0].properties["errors"] == 1

    @pytest.mark.asyncio
    async def test_run_test_with_errors(self, run_maven_tests_action, file_context):
        """Test running a Maven test with errors."""
        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/ErrorTest.java"], thoughts="Testing Maven test with errors"
        )

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify the response
        assert "errors" in result.message.lower()
        assert len(result.artifact_changes) == 1
        assert result.artifact_changes[0].properties["errors"] == 1

    @pytest.mark.asyncio
    async def test_compilation_failure(self, run_maven_tests_action, file_context, mock_environment):
        """Test Maven compilation failure is handled correctly."""
        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/SuccessTest.java"], thoughts="Testing compilation failure handling"
        )

        # Set flag to make the mock environment return a compilation failure
        mock_environment.execute.simulate_compilation_failure = True

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify the response
        assert "Maven compilation failed" in result.message
        assert "Tests cannot be run until compilation errors are fixed" in result.message
        assert "Compilation errors:" in result.message
        assert "cannot infer type arguments" in result.message
        assert "incompatible types" in result.message or "cannot find symbol" in result.message
        assert result.properties.get("fail_reason") == "compilation_failed"

        # Verify no test results were returned
        assert result.properties.get("test_results") == []

        # Verify no test files were processed
        file_context.add_test_files.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_multiple_tests(self, run_maven_tests_action, file_context):
        """Test running multiple Maven tests."""
        args = RunMavenTestsArgs(
            test_files=[
                "src/test/java/com/example/SuccessTest.java",
                "src/test/java/com/example/FailTest.java",
                "src/test/java/com/example/ErrorTest.java",
            ],
            thoughts="Testing running multiple Maven tests",
        )

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify the response
        assert "Total:" in result.message
        assert len(result.artifact_changes) == 3

    @pytest.mark.asyncio
    async def test_non_existent_file(self, run_maven_tests_action, file_context, mock_environment):
        """Test behavior when a non-existent file is provided."""
        # Override the file_exists check
        file_context.file_exists = lambda path: "NonExistentTest" not in path

        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/NonExistentTest.java"],
            thoughts="Testing non-existent test file behavior",
        )

        with patch.object(mock_environment, "read_file", side_effect=FileNotFoundError("File not found")):
            result = await run_maven_tests_action.execute(args, file_context)

        # With our implementation, this should return "Files not found" message
        assert "Files not found" in result.message
        assert result.properties.get("fail_reason") == "no_test_files"

    @pytest.mark.asyncio
    async def test_directory_instead_of_file(self, run_maven_tests_action, file_context):
        """Test behavior when a directory is provided instead of a file."""
        args = RunMavenTestsArgs(test_files=["src/test/java/com/example"], thoughts="Testing directory path behavior")

        result = await run_maven_tests_action.execute(args, file_context)

        # Should include a message about directories
        assert "Directories provided" in result.message

    @pytest.mark.asyncio
    async def test_shadow_mode(self, run_maven_tests_action, file_context):
        """Test that shadow_mode prevents running tests."""
        # Set shadow_mode to True
        file_context.shadow_mode = True

        args = RunMavenTestsArgs(
            test_files=["src/test/java/com/example/SuccessTest.java"], thoughts="Testing shadow mode check"
        )

        result = await run_maven_tests_action.execute(args, file_context)

        # Verify we get the shadow mode message
        assert "Maven tests can only be run when file_context.shadow_mode is False" in result.message
        assert result.properties.get("fail_reason") == "shadow_mode_enabled"
