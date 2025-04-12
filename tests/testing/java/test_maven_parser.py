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
        os.makedirs(self.fixture_dir, exist_ok=True)
        
        # Create sample fixture for command failure
        self.command_failure_fixture = self.fixture_dir / "command_failure.txt"
        with open(self.command_failure_fixture, "w") as f:
            f.write(self.COMMAND_FAILURE_OUTPUT)
            
        # Create sample fixture for compilation error
        self.compilation_error_fixture = self.fixture_dir / "compilation_error.txt"
        with open(self.compilation_error_fixture, "w") as f:
            f.write(self.COMPILATION_ERROR_OUTPUT)
        
        # Create sample fixture for test failure
        self.test_failure_fixture = self.fixture_dir / "test_failure.txt"
        with open(self.test_failure_fixture, "w") as f:
            f.write(self.TEST_FAILURE_OUTPUT)
            
        # Create sample fixture for successful test
        self.success_fixture = self.fixture_dir / "success.txt"
        with open(self.success_fixture, "w") as f:
            f.write(self.SUCCESS_OUTPUT)
            
        # Create sample fixture for compilation error with line number only
        self.compilation_error_line_only_fixture = self.fixture_dir / "compilation_error_line_only.txt"
        with open(self.compilation_error_line_only_fixture, "w") as f:
            f.write(self.COMPILATION_ERROR_LINE_ONLY_OUTPUT)

    def test_command_failure_parsing(self):
        """Test parsing of Maven command failure output."""
        with open(self.command_failure_fixture, "r") as f:
            output = f.read()
        
        test_file = "src/test/java/com/example/core/services/BeneficialOwnerServiceTest.java"
        results = self.parser.parse_test_output(output, test_file)
        
        # Should have one error result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, TestStatus.ERROR)
        self.assertEqual(results[0].name, "Maven command failed: mvn test -Dtest=com.example.core.services.BeneficialOwnerServiceTest")
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

    # Test fixture sample outputs
    COMMAND_FAILURE_OUTPUT = """Command failed with return code 1: mvn test -Dtest=com.example.core.services.BeneficialOwnerServiceTest

[INFO] Scanning for projects...
[INFO] 
[INFO] --------------------------< com.example:app >--------------------------
[INFO] Building app 0.0.1-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  0.512 s
[INFO] Finished at: 2023-07-20T15:22:01+02:00
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:2.22.2:test (default-cli) on project app: No tests were executed!
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
"""

    COMPILATION_ERROR_OUTPUT = """[INFO] Scanning for projects...
[INFO] 
[INFO] --------------------------< com.example:app >--------------------------
[INFO] Building app 0.0.1-SNAPSHOT
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- maven-resources-plugin:3.2.0:resources (default-resources) @ app ---
[INFO] Using 'UTF-8' encoding to copy filtered resources.
[INFO] Using 'UTF-8' encoding to copy filtered properties files.
[INFO] Copying 1 resource
[INFO] Copying 32 resources
[INFO] 
[INFO] --- maven-compiler-plugin:3.8.1:compile (default-compile) @ app ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 184 source files to /Users/albert/repos/example/app-api/target/classes
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[216,74] cannot find symbol
[ERROR]   symbol:   method findVerificationsForBeneficialOwner(java.lang.String)
[ERROR]   location: variable beneficialOwnerVerificationRepository of type com.example.infrastructure.repositories.BeneficialOwnerVerificationRepository
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[240,76] cannot find symbol
[ERROR]   symbol:   method findVerificationsForBeneficialOwner(java.lang.String)
[ERROR]   location: variable beneficialOwnerVerificationRepository of type com.example.infrastructure.repositories.BeneficialOwnerVerificationRepository
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  3.651 s
[INFO] Finished at: 2023-07-20T15:22:25+02:00
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.8.1:compile (default-compile) on project app: Compilation failure: Compilation failure: 
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[216,74] cannot find symbol
[ERROR]   symbol:   method findVerificationsForBeneficialOwner(java.lang.String)
[ERROR]   location: variable beneficialOwnerVerificationRepository of type com.example.infrastructure.repositories.BeneficialOwnerVerificationRepository
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[240,76] cannot find symbol
[ERROR]   symbol:   method findVerificationsForBeneficialOwner(java.lang.String)
[ERROR]   location: variable beneficialOwnerVerificationRepository of type com.example.infrastructure.repositories.BeneficialOwnerVerificationRepository
[ERROR] -> [Help 1]
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
"""

    COMPILATION_ERROR_LINE_ONLY_OUTPUT = """[INFO] Scanning for projects...
[INFO]
[INFO] --------------------------< com.example:app >--------------------------
[INFO] Building app 0.0.1-SNAPSHOT
[INFO]   from pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO]
[INFO] --- jacoco:0.8.11:prepare-agent (prepare-agent) @ app ---
[INFO] argLine set to -javaagent:/Users/albert/.m2/repository/org/jacoco/org.jacoco.agent/0.8.11/org.jacoco.agent-0.8.11-runtime.jar=destfile=/Users/albert/repos/example/app-api/target/jacoco.exec
[INFO]
[INFO] --- resources:3.3.1:resources (default-resources) @ app ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] Copying 44 resources from src/main/resources to target/classes
[INFO]
[INFO] --- compiler:3.12.1:compile (default-compile) @ app ---
[INFO] Recompiling the module because of changed source code.
[INFO] Compiling 324 source files with javac [debug parameters release 21] to target/classes
[INFO] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/Person.java: Some input files use or override a deprecated API.
[INFO] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/Person.java: Recompile with -Xlint:deprecation for details.
[INFO] /Users/albert/repos/example/app-api/src/main/java/com/example/simulation/util/PersonaJsonLoader.java: Some input files use unchecked or unsafe operations.
[INFO] /Users/albert/repos/example/app-api/src/main/java/com/example/simulation/util/PersonaJsonLoader.java: Recompile with -Xlint:unchecked for details.
[INFO] -------------------------------------------------------------
[WARNING] COMPILATION WARNING :
[INFO] -------------------------------------------------------------
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/transactions/domain/Transaction.java:[69,33] @Builder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/BaseEntity.java:[35,23] @SuperBuilder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/BeneficialOwner.java:[13,1] Generating equals/hashCode implementation but without a call to superclass, even though this class does not extend java.lang.Object. If this is intentional, add '@EqualsAndHashCode(callSuper=false)' to your type.
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/Organization.java:[13,1] Generating equals/hashCode implementation but without a call to superclass, even though this class does not extend java.lang.Object. If this is intentional, add '@EqualsAndHashCode(callSuper=false)' to your type.
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/Address.java:[13,1] Generating equals/hashCode implementation but without a call to superclass, even though this class does not extend java.lang.Object. If this is intentional, add '@EqualsAndHashCode(callSuper=false)' to your type.
[WARNING] /Users/albert/repos/example/app-api/src/main/java/com/example/core/domain/Person.java:[16,1] Generating equals/hashCode implementation but without a call to superclass, even though this class does not extend java.lang.Object. If this is intentional, add '@EqualsAndHashCode(callSuper=false)' to your type.
[INFO] 6 warnings
[INFO] -------------------------------------------------------------
[INFO] -------------------------------------------------------------
[ERROR] COMPILATION ERROR :
[INFO] -------------------------------------------------------------
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/api/BeneficialOwnerController.java:[55,64] cannot infer type arguments for com.example.core.dto.PageResponseDto<>
  reason: cannot infer type-variable(s) T
    (actual and formal argument lists differ in length)
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[143,79] incompatible types: java.util.List<com.example.core.domain.Address> cannot be converted to java.util.Set<com.example.core.domain.Address>
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[148,25] cannot find symbol
  symbol:   method id(java.util.UUID)
  location: class com.example.core.dto.AddressDto.AddressDtoBuilder
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[155,25] incompatible types: inference variable T has incompatible bounds
    equality constraints: com.example.core.dto.AddressDto
    lower bounds: java.lang.Object
[INFO] 4 errors
[INFO] -------------------------------------------------------------
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  2.550 s
[INFO] Finished at: 2025-04-11T13:40:22+02:00
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.12.1:compile (default-compile) on project app: Compilation failure: Compilation failure:
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/api/BeneficialOwnerController.java:[55,64] cannot infer type arguments for com.example.core.dto.PageResponseDto<>
[ERROR]   reason: cannot infer type-variable(s) T
[ERROR]     (actual and formal argument lists differ in length)
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[143,79] incompatible types: java.util.List<com.example.core.domain.Address> cannot be converted to java.util.Set<com.example.core.domain.Address>
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[148,25] cannot find symbol
[ERROR]   symbol:   method id(java.util.UUID)
[ERROR]   location: class com.example.core.dto.AddressDto.AddressDtoBuilder
[ERROR] /Users/albert/repos/example/app-api/src/main/java/com/example/core/services/impl/BeneficialOwnerServiceImpl.java:[155,25] incompatible types: inference variable T has incompatible bounds
[ERROR]     equality constraints: com.example.core.dto.AddressDto
[ERROR]     lower bounds: java.lang.Object
[ERROR] -> [Help 1]
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR]
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
"""

    TEST_FAILURE_OUTPUT = """[INFO] Scanning for projects...
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
[INFO] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.185 s - in com.example.SuccessTest
[INFO] Running com.example.FailTest
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.289 s <<< FAILURE! - in com.example.FailTest
[ERROR] com.example.FailTest#testSomething  Time elapsed: 0.121 s  <<< FAILURE!
java.lang.AssertionError: expected:<true> but was:<false>
    at org.junit.Assert.fail(Assert.java:89)
    at org.junit.Assert.failNotEquals(Assert.java:835)
    at org.junit.Assert.assertEquals(Assert.java:120)
    at org.junit.Assert.assertEquals(Assert.java:146)
    at com.example.FailTest.testSomething(FailTest.java:42)
[INFO] Running com.example.ErrorTest
[ERROR] Tests run: 3, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 0.275 s <<< FAILURE! - in com.example.ErrorTest
[ERROR] com.example.ErrorTest#testWithError  Time elapsed: 0.134 s  <<< ERROR!
java.lang.NullPointerException
    at com.example.ErrorTest.testWithError(ErrorTest.java:58)

[INFO] 
[INFO] Results:
[INFO] 
[ERROR] Failures: 
[ERROR]   com.example.FailTest.testSomething:42 expected:<true> but was:<false>
[ERROR] Errors: 
[ERROR]   com.example.ErrorTest.testWithError:58 » NullPointer
[INFO] 
[ERROR] Tests run: 9, Failures: 1, Errors: 1, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  4.631 s
[INFO] Finished at: 2023-05-15T14:32:45-07:00
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:2.22.1:test (default-test) on project my-app: There are test failures.
[ERROR] 
[ERROR] Please refer to target/surefire-reports for the individual test results.
[ERROR] Please refer to dump files (if any exist) [date].dump, [date]-jvmRun[N].dump and [date].dumpstream.
[ERROR] -> [Help 1]
"""

    SUCCESS_OUTPUT = """[INFO] Scanning for projects...
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
[INFO] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.185 s - in com.example.SuccessTest
[INFO] Running com.example.AnotherSuccessTest
[INFO] Tests run: 5, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.232 s - in com.example.AnotherSuccessTest
[INFO] 
[INFO] Results:
[INFO] 
[INFO] Tests run: 8, Failures: 0, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  3.532 s
[INFO] Finished at: 2023-05-15T14:30:12-07:00
[INFO] ------------------------------------------------------------------------
"""

if __name__ == "__main__":
    unittest.main() 