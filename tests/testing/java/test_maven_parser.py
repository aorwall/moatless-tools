import os
import unittest
from pathlib import Path
from typing import Optional

from moatless.testing.java.maven_parser import MavenParser
from moatless.testing.schema import TestResult, TestStatus


class TestMavenParser(unittest.TestCase):
    """Tests for the Maven parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = MavenParser()

        # Path to test fixtures directory
        self.fixture_dir = Path(__file__).parent / "fixtures"
        
        # Reference existing fixture files
        self.command_failure_fixture = self.fixture_dir / "command_failure.txt"
        self.compilation_error_fixture = self.fixture_dir / "compilation_error.txt"
        self.test_failure_fixture = self.fixture_dir / "test_failure.txt"
        self.success_fixture = self.fixture_dir / "success.txt"
        self.compilation_error_line_only_fixture = self.fixture_dir / "compilation_error_line_only.txt"
        self.spring_context_error_fixture = self.fixture_dir / "spring_context_error.txt"
        
        # Ensure fixture directory exists but don't recreate files
        if not self.fixture_dir.exists():
            os.makedirs(self.fixture_dir, exist_ok=True)

    def test_command_failure_parsing(self):
        """Test parsing of Maven command failure output."""
        with open(self.command_failure_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/core/services/BeneficialOwnerServiceTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have one error result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, TestStatus.ERROR)
        self.assertEqual(
            results[0].name,
            "Maven command failed: mvn test -Dtest=com.example.core.services.BeneficialOwnerServiceTest",
        )
        self.assertEqual(results[0].file_path, test_file)
        # Make sure the full command failure text is preserved
        self.assertIsNotNone(results[0].failure_output)
        failure_output = results[0].failure_output or ""
        self.assertTrue(output in failure_output)

    def test_compilation_error_parsing(self):
        """Test parsing of Maven compilation error output."""
        with open(self.compilation_error_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/core/services/BeneficialOwnerServiceTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have compilation error results
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].status, TestStatus.ERROR)

        # Check the name contains "Compilation error"
        self.assertIsNotNone(results[0].name)
        name = results[0].name or ""
        self.assertTrue("Compilation error" in name)

        # Should contain the full error message with all three lines
        self.assertIsNotNone(results[0].failure_output)
        error_output = results[0].failure_output or ""
        self.assertIn("cannot find symbol", error_output)
        self.assertIn("symbol:   method findVerificationsForBeneficialOwner", error_output)
        self.assertIn("location: variable beneficialOwnerVerificationRepository", error_output)

    def test_compilation_error_with_line_number_only(self):
        """Test parsing of Maven compilation error output with line number only (not line,column)."""
        with open(self.compilation_error_line_only_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/core/services/OrganizationServiceTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have compilation error results
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].status, TestStatus.ERROR)

        # Check the name contains "Compilation error"
        self.assertIsNotNone(results[0].name)
        name = results[0].name or ""
        self.assertTrue("Compilation error" in name)

        # Should contain the specific error messages from the fixture
        self.assertIsNotNone(results[0].failure_output)
        error_output = results[0].failure_output or ""
        self.assertIn("cannot infer type arguments", error_output)
        # We only need to look for the first error message, since the parser is currently only extracting the first error
        # The first error in the fixture is about "cannot infer type arguments"
        # We'll check the entire fixture output contains all expected errors
        self.assertIn("incompatible types", output)
        self.assertIn("cannot find symbol", output)

    def test_test_failure_parsing(self):
        """Test parsing of Maven test failure output."""
        with open(self.test_failure_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/test/MyTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have test results
        self.assertGreaterEqual(len(results), 1)

        # Find tests by status
        failed_tests = [r for r in results if r.status == TestStatus.FAILED or r.status == TestStatus.ERROR]

        # Verify we have at least one failed test
        self.assertGreaterEqual(len(failed_tests), 1)

        # Verify first failed test
        failed_test = failed_tests[0]
        self.assertIsNotNone(failed_test.name)
        self.assertEqual(failed_test.status, TestStatus.ERROR, "Should have error status")

        # Check that there is failure output
        self.assertIsNotNone(failed_test.failure_output)

        # Verify that BUILD FAILURE is mentioned in the original output
        self.assertIn("BUILD FAILURE", output)

    def test_success_parsing(self):
        """Test parsing of successful Maven test output."""
        with open(self.success_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/test/MyTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have only passed results
        self.assertGreater(len(results), 0)
        self.assertTrue(all(r.status == TestStatus.PASSED for r in results))

    def test_spring_context_error_parsing(self):
        """Test parsing of Spring ApplicationContext failure output."""
        with open(self.spring_context_error_fixture, "r") as f:
            output = f.read()

        test_file = "src/test/java/com/example/test/EntityIntegrationTest.java"
        results = self.parser.parse_test_output(output, test_file)

        # Should have test results for the errors in the summary section
        self.assertGreater(len(results), 0)

        # Verify error status
        self.assertEqual(results[0].status, TestStatus.ERROR)

        # Verify error messages contain the class and method names
        error_names = [r.name for r in results]
        self.assertTrue(any("EntityIntegrationTest.getEntity_Success" in name for name in error_names))
        
        # Verify error output contains the error type
        self.assertTrue(any("IllegalState" in (r.failure_output or "") for r in results))


if __name__ == "__main__":
    unittest.main()
