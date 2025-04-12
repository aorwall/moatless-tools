import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from moatless.feedback.maven_compilation_checker import MavenCompilationChecker
from moatless.environment.base import BaseEnvironment
from moatless.node import Node
from moatless.workspace import Workspace


@pytest.fixture
def mock_environment():
    """Create a mock environment that simulates Maven command execution."""
    mock_env = AsyncMock(spec=BaseEnvironment)
    
    # Define a helper function to generate Maven-like output based on the command
    async def execute_mock(command, fail_on_error=False):
        if "mvn --version" in command:
            return "Apache Maven 3.8.6 (84538c9988a25aec085021c365c560670ad80f63)"
            
        if "mvn compile" in command:
            # Check if we should simulate a compilation failure
            if getattr(execute_mock, "simulate_compilation_failure", False):
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
[INFO] -------------------------------------------------------------
[WARNING] COMPILATION WARNING :
[INFO] -------------------------------------------------------------
[WARNING] /home/user/projects/app/src/main/java/com/example/domain/Transaction.java:[69,33] @Builder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[WARNING] /home/user/projects/app/src/main/java/com/example/core/domain/BaseEntity.java:[35,23] @SuperBuilder will ignore the initializing expression entirely. If you want the initializing expression to serve as default, add @Builder.Default. If it is not supposed to be settable during building, make the field final.
[INFO] 2 warnings
[INFO] -------------------------------------------------------------
[INFO] -------------------------------------------------------------
[ERROR] COMPILATION ERROR :
[INFO] -------------------------------------------------------------
[ERROR] /home/user/projects/app/src/main/java/com/example/core/api/CustomerController.java:[55,64] cannot infer type arguments for com.example.core.dto.PageResponseDto<>
  reason: cannot infer type-variable(s) T
    (actual and formal argument lists differ in length)
[ERROR] /home/user/projects/app/src/main/java/com/example/core/services/impl/CustomerServiceImpl.java:[143,79] incompatible types: java.util.List<com.example.core.domain.Address> cannot be converted to java.util.Set<com.example.core.domain.Address>
[INFO] 2 errors
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
[ERROR] -> [Help 1]
"""
            # Default to successful compilation
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
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 324 source files with javac [debug parameters release 21] to target/classes
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  2.350 s
[INFO] Finished at: 2025-04-11T13:45:22+02:00
[INFO] ------------------------------------------------------------------------
"""
        
        # Default response for other commands
        return ""
        
    mock_env.execute = execute_mock
    
    return mock_env


@pytest.fixture
def workspace(mock_environment):
    """Create a workspace with mock environment."""
    workspace = MagicMock(spec=Workspace)
    workspace.environment = mock_environment
    return workspace


@pytest.fixture
def no_env_workspace():
    """Create a workspace with no environment."""
    workspace = MagicMock(spec=Workspace)
    workspace.environment = None
    return workspace


@pytest.fixture
def node():
    """Create a basic test node."""
    # Create a root node using the helper method
    return Node.create_root(user_message="Test message")


@pytest_asyncio.fixture
async def checker(workspace):
    """Create a MavenCompilationChecker with workspace."""
    checker = MavenCompilationChecker()
    await checker.initialize(workspace)
    return checker


class TestMavenCompilationChecker:
    
    @pytest.mark.asyncio
    async def test_successful_compilation(self, checker, node):
        """Test that no feedback is returned for successful compilation."""
        # Use default mock behavior (successful compilation)
        result = await checker.generate_feedback(node)
        
        # Should return None for successful compilation
        assert result is None
        
    @pytest.mark.asyncio
    async def test_compilation_failure(self, checker, node, mock_environment):
        """Test that proper feedback is generated for compilation failure."""
        # Set flag to make the mock environment return a compilation failure
        mock_environment.execute.simulate_compilation_failure = True
        
        result = await checker.generate_feedback(node)
        
        # Check that we have feedback
        assert result is not None
        assert result.feedback is not None
        
        # Verify the content of the feedback
        assert "Maven compilation failed" in result.feedback
        assert "Compilation Errors" in result.feedback
        assert "CustomerController.java" in result.feedback
        assert "cannot infer type arguments" in result.feedback
        assert "incompatible types" in result.feedback
        
    @pytest.mark.asyncio
    async def test_maven_not_installed(self, workspace, node):
        """Test handling of missing Maven."""
        # Create a new checker for this test
        checker = MavenCompilationChecker()
        
        # Make Maven version check fail
        async def fail_execute(command, fail_on_error=False):
            if "mvn --version" in command:
                raise Exception("Command 'mvn' not found")
            return ""
            
        # Replace the execute method before initialization
        workspace.environment.execute = fail_execute
        
        # Now we expect a RuntimeError to be raised during initialization
        with pytest.raises(RuntimeError) as excinfo:
            await checker.initialize(workspace)
        
        # Check that the error message contains the expected information
        assert "Maven does not appear to be installed" in str(excinfo.value)
        assert "Command 'mvn' not found" in str(excinfo.value)
        
    @pytest.mark.asyncio
    async def test_maven_execution_error(self, checker, node, workspace):
        """Test handling of Maven execution errors."""
        # Make Maven compile command fail with an execution error
        async def fail_execute(command, fail_on_error=False):
            if "mvn --version" in command:
                return "Apache Maven 3.8.6"
            if "mvn compile" in command:
                raise Exception("Failed to execute Maven: error code 1")
            return ""
            
        # Replace the execute method before running generate_feedback
        workspace.environment.execute = fail_execute
        
        # Now we expect a RuntimeError to be raised
        with pytest.raises(RuntimeError) as excinfo:
            await checker.generate_feedback(node)
        
        # Check that the error message contains the expected information
        assert "Failed to execute Maven compilation" in str(excinfo.value)
        assert "error code 1" in str(excinfo.value)
        
    @pytest.mark.asyncio
    async def test_workspace_not_set(self):
        """Test handling of missing workspace."""
        checker = MavenCompilationChecker()
        node = Node.create_root(user_message="Test message")
        
        result = await checker.generate_feedback(node)
        
        # Should return None when workspace is not set
        assert result is None
        
    @pytest.mark.asyncio
    async def test_environment_not_set(self, no_env_workspace, node):
        """Test handling of missing environment."""
        checker = MavenCompilationChecker()
        
        # We should not try to initialize with a null environment
        try:
            await checker.initialize(no_env_workspace)
            assert False, "Expected ValueError not raised"
        except ValueError as e:
            assert "Environment is required" in str(e)
            
        # If initialization fails, the generate_feedback should return None
        result = await checker.generate_feedback(node)
        assert result is None 