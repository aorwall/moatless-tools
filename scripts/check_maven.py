#!/usr/bin/env python3
"""
This script checks if Maven is installed on the system and 
tests the Maven parser with sample Maven output.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from moatless.testing.java.maven_parser import MavenParser
from moatless.testing.schema import TestStatus


def check_maven_installed():
    """Check if Maven is installed on the system."""
    try:
        result = subprocess.run(
            ["mvn", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print("Maven is installed:")
            print(result.stdout.splitlines()[0])
            return True
        else:
            print("Maven command failed:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("Maven is not installed or not found in PATH")
        return False


def test_maven_parser():
    """Test the Maven parser with sample Maven output."""
    print("\nTesting Maven parser with sample output...")
    
    # Sample Maven output
    sample_output = """
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
"""
    
    parser = MavenParser()
    test_results = parser.parse_test_output(sample_output)
    
    # Print the test results
    print(f"Found {len(test_results)} test results:")
    
    # Count by status
    passed = sum(1 for r in test_results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
    errors = sum(1 for r in test_results if r.status == TestStatus.ERROR)
    skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
    
    print(f"Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
    
    # Print details of each test
    for i, result in enumerate(test_results):
        print(f"\nTest Result {i+1}:")
        print(f"  Status: {result.status}")
        print(f"  Name: {result.name}")
        print(f"  File Path: {result.file_path}")
        print(f"  Method: {result.method}")
        if result.failure_output:
            print(f"  Failure Output: {result.failure_output[:100]}...")


def test_real_maven_project():
    """Test the Maven parser with real output from tm-api project."""
    print("\nTesting Maven parser with real output from tm-api project...")
    
    # Test files from the user query
    test_files = [
        "src/test/java/se/frankpenny/tm/core/services/BeneficialOwnerServiceTest.java",
        "src/test/java/se/frankpenny/tm/core/services/OrganizationServiceTest.java"
    ]
    
    # Path to the tm-api project
    tm_api_path = "/Users/albert/repos/fp/tm-api"
    
    if not os.path.exists(tm_api_path):
        print(f"Error: Project path {tm_api_path} does not exist.")
        return
    
    parser = MavenParser()
    
    # Process each test file
    for test_file in test_files:
        print(f"\nProcessing test file: {test_file}")
        
        # Extract class name from file path
        file_without_ext = test_file[:-5]  # Remove ".java"
        
        # Handle common source directories
        for prefix in ["src/test/java/", "src/main/java/", "src/it/java/"]:
            if file_without_ext.startswith(prefix):
                file_without_ext = file_without_ext[len(prefix):]
                break
        
        # Convert / to . for package name
        class_name = file_without_ext.replace("/", ".")
        
        # Build Maven test command
        command = f"cd {tm_api_path} && mvn test -Dtest={class_name}"
        print(f"Running command: {command}")
        
        try:
            # Run the Maven test
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Capture output regardless of success/failure
            # Combine stdout and stderr for better error detection
            output = process.stdout
            if process.stderr:
                if output:
                    output += "\n" + process.stderr
                else:
                    output = process.stderr
            
            # Check for failure
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                if not output:
                    output = f"Command failed with return code {process.returncode}: mvn test -Dtest={class_name}"
            
            # Parse the test output
            test_results = parser.parse_test_output(output, test_file)
            
            # Print the test results
            print(f"Found {len(test_results)} test results:")
            
            # Count by status
            passed = sum(1 for r in test_results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
            errors = sum(1 for r in test_results if r.status == TestStatus.ERROR)
            skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
            
            print(f"Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}")
            
            # Print details of each test
            for i, result in enumerate(test_results):
                print(f"\nTest Result {i+1}:")
                print(f"  Status: {result.status}")
                print(f"  Name: {result.name}")
                print(f"  File Path: {result.file_path}")
                print(f"  Method: {result.method}")
                if result.failure_output:
                    print(f"  Failure Output: {result.failure_output[:200]}...")
            
            # Special debug output to see what's being parsed
            if len(test_results) == 0:
                print("\nNo test results found. Here's a sample of the output to debug:")
                print("-" * 50)
                print(output[:500] + "..." if len(output) > 500 else output)
                print("-" * 50)
        
        except Exception as e:
            print(f"Error running Maven test: {str(e)}")


def main():
    """Main function."""
    print("Maven Test Runner and Parser Checker")
    print("====================================")
    
    # Check if Maven is installed
    maven_installed = check_maven_installed()
    
    # Test the Maven parser with sample output
    test_maven_parser()
    
    # Test with real output if Maven is installed
    if maven_installed:
        test_real_maven_project()
        print("\nYou can use the RunMavenTests action to run Maven tests.")
    else:
        print("\nMaven is not installed. You need to install Maven to run actual tests.")
        print("However, the parser should still work with logs from Maven runs.")


if __name__ == "__main__":
    main() 