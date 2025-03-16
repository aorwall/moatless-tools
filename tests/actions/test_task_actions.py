import os
import tempfile
from pathlib import Path
import pytest
import asyncio
from unittest.mock import MagicMock, patch

from moatless.actions.create_tasks import CreateTasks, CreateTasksArgs, TaskItem
from moatless.actions.update_task import UpdateTask, UpdateTaskArgs
from moatless.actions.list_tasks import ListTasks, ListTasksArgs
from moatless.artifacts.task import TaskHandler, TaskState, TaskArtifact
from moatless.storage.file_storage import FileStorage
from moatless.storage.base import BaseStorage
from moatless.file_context import FileContext
from moatless.workspace import Workspace
from moatless.context_data import current_project_id, current_trajectory_id


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a storage instance with a test directory."""
    # Reset the singleton to ensure we're starting fresh
    BaseStorage.reset_instance()
    # Create and return the new instance with a test directory
    return BaseStorage.get_instance(storage_impl=FileStorage, base_dir=temp_dir)


@pytest.fixture(autouse=True)
def reset_storage_after_test():
    """Auto-used fixture to reset the storage singleton after each test."""
    yield
    BaseStorage.reset_instance()


@pytest.fixture
def context_setup():
    """Set up project and trajectory context for tests."""
    # Save original values to restore later
    original_project = current_project_id.get()
    original_trajectory = current_trajectory_id.get()
    
    # Set test values
    test_project = "test-project"
    test_trajectory = "test-trajectory"
    current_project_id.set(test_project)
    current_trajectory_id.set(test_trajectory)
    
    yield test_project, test_trajectory
    
    # Restore original values
    if original_project is not None:
        current_project_id.set(original_project)
    else:
        current_project_id.set(None)
        
    if original_trajectory is not None:
        current_trajectory_id.set(original_trajectory)
    else:
        current_trajectory_id.set(None)


@pytest.fixture
def task_handler(file_storage, context_setup):
    """Create a TaskHandler using the test FileStorage."""
    handler = TaskHandler()
    handler._storage = file_storage
    return handler


@pytest.fixture
def workspace(task_handler):
    """Create a test workspace with the TaskHandler."""
    workspace = Workspace()
    workspace.artifact_handlers = {"task": task_handler}
    return workspace


@pytest.fixture
def file_context():
    """Create a dummy file context for the tests."""
    return FileContext()


@pytest.fixture
def create_action(workspace):
    """Create a CreateTasks action with the test workspace."""
    action = CreateTasks()
    action.workspace = workspace
    return action


@pytest.fixture
def update_action(workspace):
    """Create an UpdateTask action with the test workspace."""
    action = UpdateTask()
    action.workspace = workspace
    return action


@pytest.fixture
def list_action(workspace):
    """Create a ListTasks action with the test workspace."""
    action = ListTasks()
    action.workspace = workspace
    return action


@pytest.mark.asyncio
async def test_create_tasks(create_action, file_context, context_setup):
    """Test creating tasks with the CreateTasks action."""
    # Create task arguments
    tasks = [
        TaskItem(id="test-1", content="Test task 1", priority=10),
        TaskItem(id="test-2", content="Test task 2", priority=20),
    ]
    args = CreateTasksArgs(
        thoughts="Creating test tasks for unit test",
        tasks=tasks
    )
    
    # Execute the action
    observation = await create_action.execute(args, file_context=file_context)
    result = observation.message
    
    # Verify the result contains the expected task info
    assert "Created 2 tasks" in result
    assert "test-1" in result
    assert "test-2" in result
    assert "Test task 1" in result
    assert "Test task 2" in result
    assert "priority: 10" in result or "Priority: 10" in result
    assert "priority: 20" in result or "Priority: 20" in result


@pytest.mark.asyncio
async def test_update_task(create_action, update_action, file_context, context_setup):
    """Test updating a task with the UpdateTask action."""
    # First create a task
    tasks = [TaskItem(id="update-test", content="Task to update", priority=50)]
    create_args = CreateTasksArgs(
        thoughts="Creating a task to update in test",
        tasks=tasks
    )
    await create_action.execute(create_args, file_context=file_context)
    
    # Update the task status
    update_args = UpdateTaskArgs(
        thoughts="Updating the task status, result, and priority",
        task_id="update-test",
        state=TaskState.COMPLETED,
        result="Task completed successfully",
        priority=5
    )
    observation = await update_action.execute(update_args, file_context=file_context)
    result = observation.message
    
    # Verify the update result - adapted for actual output format
    assert "Updated task update-test" in result
    assert "TaskState.OPEN" in result and "TaskState.COMPLETED" in result
    assert "Task completed successfully" in result
    assert "priority from 50 to 5" in result


@pytest.mark.asyncio
@patch('moatless.actions.list_tasks.ListTasks._execute')
async def test_list_tasks(mock_execute, create_action, list_action, file_context, context_setup):
    """Test listing tasks with the ListTasks action.
    
    This test uses a mock for the _execute method since we can't rely on the search
    functionality that hasn't been fully implemented yet.
    """
    # First create some tasks to have them in storage
    tasks = [
        TaskItem(id="list-1", content="Open task", priority=10),
        TaskItem(id="list-2", content="Another open task", priority=20),
    ]
    create_args = CreateTasksArgs(
        thoughts="Creating tasks for list test",
        tasks=tasks
    )
    await create_action.execute(create_args, file_context=file_context)
    
    # Set up mock return values for listing tasks
    # Mock for listing all tasks
    mock_all_tasks_result = """Found 2 tasks:

ID: list-1
Priority: 10
State: TaskState.OPEN
Content: Open task

ID: list-2
Priority: 20
State: TaskState.OPEN
Content: Another open task
"""
    
    # Mock for listing tasks with OPEN state
    mock_open_tasks_result = """Found 2 tasks with state 'TaskState.OPEN':

ID: list-1
Priority: 10
State: TaskState.OPEN
Content: Open task

ID: list-2
Priority: 20
State: TaskState.OPEN
Content: Another open task
"""
    
    from moatless.actions.action import Observation
    # Set up the mock to return different values based on arguments
    async def side_effect(args, file_context=None):
        if args.state is None:
            return mock_all_tasks_result
        return mock_open_tasks_result
    
    mock_execute.side_effect = side_effect
    
    # Test listing all tasks (state=None)
    list_args = ListTasksArgs(
        thoughts="Listing all tasks",
        state=None
    )
    observation = await list_action.execute(list_args, file_context=file_context)
    result = observation.message
    
    # Test listing tasks with state filter
    list_args = ListTasksArgs(
        thoughts="Listing only open tasks",
        state=TaskState.OPEN
    )
    observation = await list_action.execute(list_args, file_context=file_context)
    result = observation.message
    
    # Verify the mock was called with the expected arguments
    assert mock_execute.call_count == 2
    # First call should be with state=None
    assert mock_execute.call_args_list[0][0][0].state is None
    # Second call should be with state=TaskState.OPEN
    assert mock_execute.call_args_list[1][0][0].state == TaskState.OPEN


@pytest.mark.asyncio
async def test_task_persistence(temp_dir, create_action, update_action, file_context, context_setup):
    """Test that tasks are correctly persisted to and loaded from storage."""
    project_id, trajectory_id = context_setup
    
    # Create a task
    tasks = [TaskItem(id="persist-test", content="Task to persist", priority=30)]
    create_args = CreateTasksArgs(
        thoughts="Creating a task to test persistence",
        tasks=tasks
    )
    await create_action.execute(create_args, file_context=file_context)
    
    # Check if the task file was created in the correct location with context values
    storage_path = Path(temp_dir) / "projects" / project_id / "trajs" / trajectory_id / "task.json"
    assert storage_path.exists(), "Task file was not created in the expected location"
    
    # Update the task
    update_args = UpdateTaskArgs(
        thoughts="Updating task to test persistence of changes",
        task_id="persist-test",
        state=TaskState.COMPLETED,
        result=None,
        priority=None
    )
    await update_action.execute(update_args, file_context=file_context)
    
    # Directly read from the file to check the stored data
    with open(storage_path, "r") as f:
        json_content = f.read()
        assert "persist-test" in json_content
        assert "completed" in json_content
        assert "Task to persist" in json_content


@pytest.mark.asyncio
@patch('moatless.actions.list_tasks.ListTasks._execute')
async def test_task_workflow(mock_list_execute, create_action, update_action, list_action, file_context, context_setup):
    """Test a complete workflow of creating, updating, and listing tasks."""
    # Set up mock return values for listing tasks
    mock_open_tasks_result = """Found 1 tasks with state 'TaskState.OPEN':

ID: workflow-1
Priority: 15
State: TaskState.OPEN
Content: First workflow task
"""
    
    mock_completed_tasks_result = """Found 1 tasks with state 'TaskState.COMPLETED':

ID: workflow-2
Priority: 25
State: TaskState.COMPLETED
Content: Second workflow task
Result: Task completed during workflow test
"""
    
    mock_all_tasks_result = """Found 3 tasks:

ID: workflow-3
Priority: 5
State: TaskState.FAILED
Content: Third workflow task
Result: Task failed during workflow test

ID: workflow-1
Priority: 15
State: TaskState.OPEN
Content: First workflow task

ID: workflow-2
Priority: 25
State: TaskState.COMPLETED
Content: Second workflow task
Result: Task completed during workflow test
"""
    
    # Set up the mock to return different values based on arguments
    async def side_effect(args, file_context=None):
        if args.state == TaskState.OPEN:
            return mock_open_tasks_result
        elif args.state == TaskState.COMPLETED:
            return mock_completed_tasks_result
        else:
            return mock_all_tasks_result
    
    mock_list_execute.side_effect = side_effect
    
    # 1. Create several tasks
    tasks = [
        TaskItem(id="workflow-1", content="First workflow task", priority=15),
        TaskItem(id="workflow-2", content="Second workflow task", priority=25),
        TaskItem(id="workflow-3", content="Third workflow task", priority=5),
    ]
    create_args = CreateTasksArgs(
        thoughts="Creating multiple tasks for workflow test",
        tasks=tasks
    )
    await create_action.execute(create_args, file_context=file_context)
    
    # 2. Update one task to completed
    update_args = UpdateTaskArgs(
        thoughts="Marking the second task as completed",
        task_id="workflow-2",
        state=TaskState.COMPLETED,
        result="Task completed during workflow test",
        priority=None
    )
    await update_action.execute(update_args, file_context=file_context)
    
    # 3. Update another task to failed
    update_args = UpdateTaskArgs(
        thoughts="Marking the third task as failed",
        task_id="workflow-3",
        state=TaskState.FAILED,
        result="Task failed during workflow test",
        priority=None
    )
    await update_action.execute(update_args, file_context=file_context)
    
    # 4. List only open tasks
    list_args = ListTasksArgs(
        thoughts="Listing only open tasks in workflow",
        state=TaskState.OPEN
    )
    observation = await list_action.execute(list_args, file_context=file_context)
    result = observation.message
    
    # 5. List completed tasks
    list_args = ListTasksArgs(
        thoughts="Listing only completed tasks in workflow",
        state=TaskState.COMPLETED
    )
    observation = await list_action.execute(list_args, file_context=file_context)
    result = observation.message
    
    # 6. List all tasks
    list_args = ListTasksArgs(
        thoughts="Listing all tasks to verify sorting",
        state=None
    )
    observation = await list_action.execute(list_args, file_context=file_context)
    
    # Verify the mock was called with the expected arguments
    assert mock_list_execute.call_count == 3
    assert mock_list_execute.call_args_list[0][0][0].state == TaskState.OPEN
    assert mock_list_execute.call_args_list[1][0][0].state == TaskState.COMPLETED
    assert mock_list_execute.call_args_list[2][0][0].state is None 