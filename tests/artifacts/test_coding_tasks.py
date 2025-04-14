import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from moatless.actions.add_coding_tasks import AddCodingTasks, AddCodingTasksArgs, CodingTaskItem
from moatless.actions.remove_coding_tasks import (
    RemoveCodingTasks,
    RemoveCodingTasksArgs,
)
from moatless.actions.finish_coding_tasks import FinishCodingTasks, FinishCodingTasksArgs
from moatless.artifacts.coding_task import (
    CodingTaskHandler,
    FileLocation,
    FileRelationType,
)

from moatless.context_data import current_project_id, current_trajectory_id
from moatless.file_context import FileContext
from moatless.storage.file_storage import FileStorage
from moatless.workspace import Workspace


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a storage instance with a test directory."""
    return FileStorage(base_dir=temp_dir)


@pytest.fixture(autouse=True)
def mock_settings_get_storage(file_storage):
    """Mock the settings.get_storage function to return our test storage."""

    async def mock_get_storage():
        return file_storage

    with patch("moatless.settings.get_storage", mock_get_storage):
        yield


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
def coding_task_handler(file_storage):
    """Create a CodingTaskHandler using the test FileStorage."""
    handler = CodingTaskHandler()
    handler._storage = file_storage
    return handler


@pytest.fixture
def workspace(coding_task_handler):
    """Create a test workspace with the CodingTaskHandler."""
    workspace = Workspace()
    workspace.artifact_handlers = {"coding_task": coding_task_handler}
    return workspace


@pytest.fixture
def file_context():
    """Create a dummy file context for the tests."""
    return FileContext()


@pytest.fixture
def add_action(workspace):
    """Create an AddCodingTasks action with the test workspace."""
    action = AddCodingTasks()
    action.workspace = workspace
    return action


@pytest.fixture
def remove_action(workspace):
    """Create a RemoveCodingTasks action with the test workspace."""
    action = RemoveCodingTasks()
    action.workspace = workspace
    return action


@pytest.fixture
def finish_action(workspace):
    """Create a FinishCodingTasks action with the test workspace."""
    action = FinishCodingTasks()
    action.workspace = workspace
    return action


@pytest.fixture(autouse=True)
def mock_settings_get_runner():
    """Mock the settings.get_runner function to return a mock runner."""
    mock_runner = AsyncMock()

    async def mock_get_runner():
        return mock_runner

    with patch("moatless.settings.get_runner", mock_get_runner):
        yield mock_runner


@pytest.mark.asyncio
async def test_add_coding_tasks(add_action, file_context, context_setup):
    """Test adding coding tasks with the AddCodingTasks action."""
    # Create task arguments with file locations
    tasks = [
        CodingTaskItem(
            id="coding-1",
            title="Fix authentication bug",
            instructions="Fix the authentication bug in the login controller",
            related_files=[
                FileLocation(
                    file_path="src/controllers/auth.js",
                    start_line=15,
                    end_line=25,
                    relation_type=FileRelationType.UPDATE,
                ),
                FileLocation(file_path="src/models/user.js", relation_type=FileRelationType.REFERENCE),
            ],
            priority=10,
        ),
        CodingTaskItem(
            id="coding-2",
            title="Add new API endpoint",
            instructions="Create a new API endpoint for user profiles",
            related_files=[FileLocation(file_path="src/routes/api.js", relation_type=FileRelationType.UPDATE)],
            priority=20,
        ),
    ]
    args = AddCodingTasksArgs(thoughts="Creating coding tasks for test", tasks=tasks)

    # Execute the action
    observation = await add_action.execute(args, file_context=file_context)
    result = observation.message

    # Verify the result contains the expected task info with checkboxes
    assert "Added 2 coding tasks" in result
    assert "[ ] coding-1 - Fix authentication bug" in result
    assert "[ ] coding-2 - Add new API endpoint" in result


@pytest.mark.asyncio
async def test_finish_coding_tasks(add_action, finish_action, file_context, context_setup):
    """Test finishing coding tasks with the FinishCodingTasks action."""
    # First, add some tasks
    tasks = [
        CodingTaskItem(id="finish-1", title="Task to finish 1", instructions="This task will be finished", priority=10),
        CodingTaskItem(
            id="finish-2", title="Task to finish 2", instructions="This task will also be finished", priority=20
        ),
        CodingTaskItem(
            id="finish-3", title="Task to remain open", instructions="This task will remain open", priority=30
        ),
    ]
    add_args = AddCodingTasksArgs(thoughts="Creating tasks for finish test", tasks=tasks)
    await add_action.execute(add_args, file_context=file_context)

    # Now finish some tasks
    finish_args = FinishCodingTasksArgs(
        thoughts="Finishing selected tasks", task_ids=["finish-1", "finish-2", "nonexistent-task"]
    )

    observation = await finish_action.execute(finish_args, file_context=file_context)
    result = observation.message

    # Verify the result
    assert "Completed 2 coding tasks" in result
    assert "finish-1" in result
    assert "finish-2" in result
    assert "Tasks not found: nonexistent-task" in result

    # Verify checkboxes in output
    assert "[x] finish-1 - Task to finish 1" in result
    assert "[x] finish-2 - Task to finish 2" in result
    assert "[ ] finish-3 - Task to remain open" in result


@pytest.mark.asyncio
async def test_remove_coding_tasks(add_action, remove_action, file_context, context_setup):
    """Test removing coding tasks with the RemoveCodingTasks action."""
    # First, add some tasks
    tasks = [
        CodingTaskItem(id="remove-1", title="Task to remove 1", instructions="This task will be removed", priority=10),
        CodingTaskItem(
            id="remove-2", title="Task to remove 2", instructions="This task will also be removed", priority=20
        ),
        CodingTaskItem(id="keep-3", title="Task to keep", instructions="This task will not be removed", priority=30),
    ]
    add_args = AddCodingTasksArgs(thoughts="Creating tasks for remove test", tasks=tasks)
    await add_action.execute(add_args, file_context=file_context)

    # Now remove some tasks
    remove_args = RemoveCodingTasksArgs(
        thoughts="Removing selected tasks", task_ids=["remove-1", "remove-2", "nonexistent-task"]
    )

    observation = await remove_action.execute(remove_args, file_context=file_context)
    result = observation.message

    # Verify the result
    assert "Removed 2 coding tasks" in result
    assert "Tasks not found: nonexistent-task" in result

    # Verify remaining tasks
    assert "keep-3" in result
    assert "remove-1" not in result or "remove-1 - Task to remove 1" not in result
    assert "remove-2" not in result or "remove-2 - Task to remove 2" not in result


@pytest.mark.asyncio
async def test_task_persistence(temp_dir, context_setup, add_action, finish_action, file_context):
    """Test that coding tasks are correctly persisted to and loaded from storage."""
    project_id, trajectory_id = context_setup

    # Create a task with file locations
    tasks = [
        CodingTaskItem(
            id="persist-test",
            title="Task to persist",
            instructions="Testing persistence of coding tasks",
            related_files=[
                FileLocation(file_path="src/main.js", start_line=10, end_line=20, relation_type=FileRelationType.UPDATE)
            ],
            priority=30,
        )
    ]
    add_args = AddCodingTasksArgs(thoughts="Creating a task to test persistence", tasks=tasks)
    await add_action.execute(add_args, file_context=file_context)

    # Check if the task file was created in the correct location with context values
    storage_path = Path(temp_dir) / "projects" / project_id / "trajs" / trajectory_id / "coding_task.json"
    assert storage_path.exists(), "Coding task file was not created in the expected location"

    # Finish the task
    finish_args = FinishCodingTasksArgs(
        thoughts="Finishing task to test persistence of changes", task_ids=["persist-test"]
    )
    await finish_action.execute(finish_args, file_context=file_context)

    # Directly read from the file to check the stored data
    with open(storage_path, "r") as f:
        json_content = f.read()
        assert "persist-test" in json_content
        assert "Task to persist" in json_content
        assert "completed" in json_content
        assert "src/main.js" in json_content
        assert "update" in json_content


@pytest.mark.asyncio
async def test_complete_coding_task_workflow(add_action, finish_action, remove_action, file_context, context_setup):
    """Test a complete workflow of adding, finishing, and removing coding tasks."""
    # 1. Add several tasks with different priorities and file references
    tasks = [
        CodingTaskItem(
            id="workflow-1",
            title="High priority task",
            instructions="This is a high priority task",
            related_files=[FileLocation(file_path="src/critical.js", relation_type=FileRelationType.UPDATE)],
            priority=10,
        ),
        CodingTaskItem(
            id="workflow-2",
            title="Medium priority task",
            instructions="This is a medium priority task",
            related_files=[FileLocation(file_path="src/important.js", relation_type=FileRelationType.CREATE)],
            priority=50,
        ),
        CodingTaskItem(
            id="workflow-3",
            title="Low priority task",
            instructions="This is a low priority task",
            related_files=[FileLocation(file_path="src/utils.js", relation_type=FileRelationType.REFERENCE)],
            priority=100,
        ),
    ]
    add_args = AddCodingTasksArgs(thoughts="Creating multiple tasks for workflow test", tasks=tasks)
    observation = await add_action.execute(add_args, file_context=file_context)

    # Verify all tasks were added with correct sorting by priority
    result = observation.message
    assert "workflow-1" in result
    assert "workflow-2" in result
    assert "workflow-3" in result

    # 2. Complete the medium priority task
    finish_args = FinishCodingTasksArgs(thoughts="Completing medium priority task", task_ids=["workflow-2"])
    observation = await finish_action.execute(finish_args, file_context=file_context)

    # Verify task was marked as completed
    result = observation.message
    assert "[x] workflow-2 - Medium priority task" in result
    assert "[ ] workflow-1 - High priority task" in result
    assert "[ ] workflow-3 - Low priority task" in result

    # 3. Remove the low priority task
    remove_args = RemoveCodingTasksArgs(thoughts="Removing low priority task", task_ids=["workflow-3"])
    observation = await remove_action.execute(remove_args, file_context=file_context)

    # Verify task was removed
    result = observation.message
    assert "workflow-3" not in result or "workflow-3 - Low priority task" not in result
    assert "workflow-1" in result
    assert "workflow-2" in result
