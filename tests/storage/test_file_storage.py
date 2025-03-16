import json
import pytest
import os
import tempfile
from pathlib import Path
import shutil

import asyncio
from unittest.mock import patch, MagicMock

from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a FileStorage instance with a temporary directory."""
    return FileStorage(base_dir=temp_dir)


@pytest.mark.asyncio
async def test_read_write_basic(file_storage):
    """Test basic read/write operations."""
    test_key = "test_key"
    test_data = {"hello": "world", "number": 42}
    
    # Write data
    await file_storage.write(test_key, test_data)
    
    # Verify file exists
    assert await file_storage.exists(test_key)
    
    # Read data back
    read_data = await file_storage.read(test_key)
    assert read_data == test_data
    
    # Update data
    updated_data = {"hello": "updated", "number": 100}
    await file_storage.write(test_key, updated_data)
    
    # Read updated data
    read_data = await file_storage.read(test_key)
    assert read_data == updated_data


@pytest.mark.asyncio
async def test_delete(file_storage):
    """Test delete operation."""
    test_key = "test_delete"
    test_data = {"to_be_deleted": True}
    
    # Write data
    await file_storage.write(test_key, test_data)
    assert await file_storage.exists(test_key)
    
    # Delete data
    await file_storage.delete(test_key)
    
    # Verify it's gone
    assert not await file_storage.exists(test_key)
    
    # Deleting a non-existent key should raise KeyError
    with pytest.raises(KeyError):
        await file_storage.delete("nonexistent_key")


@pytest.mark.asyncio
async def test_hierarchical_keys(file_storage):
    """Test hierarchical keys with slashes."""
    # Test with hierarchical key
    test_key = "folder/subfolder/test_key"
    test_data = {"nested": True}
    
    # Write data
    await file_storage.write(test_key, test_data)
    
    # Verify file exists
    assert await file_storage.exists(test_key)
    
    # Read data back
    read_data = await file_storage.read(test_key)
    assert read_data == test_data
    
    # Ensure the directory structure was created
    path = file_storage._get_path(test_key)
    assert path.exists()
    assert path.parent.name == "subfolder"
    assert path.parent.parent.name == "folder"


@pytest.mark.asyncio
async def test_project_operations(file_storage):
    """Test project-level operations."""
    # Set up test data
    project_id = "test_project"
    test_key = "project_data"
    test_data = {"project": True, "name": "Test Project"}
    
    # Write to project
    await file_storage.write_to_project(test_key, test_data, project_id)
    
    # Verify data exists
    assert await file_storage.exists_in_project(test_key, project_id)
    
    # Read from project
    read_data = await file_storage.read_from_project(test_key, project_id)
    assert read_data == test_data
    
    # Delete from project
    await file_storage.delete_from_project(test_key, project_id)
    assert not await file_storage.exists_in_project(test_key, project_id)


@pytest.mark.asyncio
async def test_trajectory_operations(file_storage):
    """Test trajectory-level operations."""
    # Set up test data
    project_id = "test_project"
    trajectory_id = "test_trajectory"
    test_key = "trajectory_data"
    test_data = {"trajectory": True, "name": "Test Trajectory"}
    
    # Write to trajectory
    await file_storage.write_to_trajectory(test_key, test_data, project_id, trajectory_id)
    
    # Verify data exists
    assert await file_storage.exists_in_trajectory(test_key, project_id, trajectory_id)
    
    # Read from trajectory
    read_data = await file_storage.read_from_trajectory(test_key, project_id, trajectory_id)
    assert read_data == test_data
    
    # Delete from trajectory
    await file_storage.delete_from_trajectory(test_key, project_id, trajectory_id)
    assert not await file_storage.exists_in_trajectory(test_key, project_id, trajectory_id)


@pytest.mark.asyncio
async def test_shorthand_methods(file_storage):
    """Test shorthand methods (p_read, t_write, etc)."""
    # Project shorthands
    project_id = "shorthand_project"
    p_key = "p_data"
    p_data = {"shorthand": "project"}
    
    await file_storage.p_write(p_key, p_data, project_id)
    assert await file_storage.p_exists(p_key, project_id)
    p_read_data = await file_storage.p_read(p_key, project_id)
    assert p_read_data == p_data
    await file_storage.p_delete(p_key, project_id)
    assert not await file_storage.p_exists(p_key, project_id)
    
    # Trajectory shorthands
    trajectory_id = "shorthand_trajectory"
    t_key = "t_data"
    t_data = {"shorthand": "trajectory"}
    
    await file_storage.t_write(t_key, t_data, project_id, trajectory_id)
    assert await file_storage.t_exists(t_key, project_id, trajectory_id)
    t_read_data = await file_storage.t_read(t_key, project_id, trajectory_id)
    assert t_read_data == t_data
    await file_storage.t_delete(t_key, project_id, trajectory_id)
    assert not await file_storage.t_exists(t_key, project_id, trajectory_id)


@pytest.mark.asyncio
async def test_list_keys(file_storage):
    """Test listing keys with different prefixes."""
    # Create some test data
    await file_storage.write("key1", {"id": 1})
    await file_storage.write("key2", {"id": 2})
    await file_storage.write("prefix/key3", {"id": 3})
    await file_storage.write("prefix/key4", {"id": 4})
    await file_storage.write("prefix/subprefix/key5", {"id": 5})
    
    # List all keys
    all_keys = await file_storage.list_keys()
    assert len(all_keys) == 5
    assert "key1" in all_keys
    assert "key2" in all_keys
    assert "prefix/key3" in all_keys
    assert "prefix/key4" in all_keys
    assert "prefix/subprefix/key5" in all_keys
    
    # List with prefix
    prefix_keys = await file_storage.list_keys("prefix")
    assert len(prefix_keys) == 3
    assert "prefix/key3" in prefix_keys
    assert "prefix/key4" in prefix_keys
    assert "prefix/subprefix/key5" in prefix_keys
    
    # List with deeper prefix
    subprefix_keys = await file_storage.list_keys("prefix/subprefix")
    assert len(subprefix_keys) == 1
    assert "prefix/subprefix/key5" in subprefix_keys


@pytest.mark.asyncio
async def test_list_projects(file_storage):
    """Test listing projects."""
    # Create some project data
    await file_storage.write_to_project("data1", {"test": 1}, "project1")
    await file_storage.write_to_project("data2", {"test": 2}, "project2")
    await file_storage.write_to_project("data3", {"test": 3}, "project3")
    
    # List projects
    projects = await file_storage.list_projects()
    assert len(projects) == 3
    assert "project1" in projects
    assert "project2" in projects
    assert "project3" in projects


@pytest.mark.asyncio
async def test_list_trajectories(file_storage):
    """Test listing trajectories within a project."""
    # Create test project with trajectories
    project_id = "traj_project"
    await file_storage.write_to_trajectory("data1", {"traj": 1}, project_id, "traj1")
    await file_storage.write_to_trajectory("data2", {"traj": 2}, project_id, "traj2")
    await file_storage.write_to_trajectory("data3", {"traj": 3}, project_id, "traj3")
    
    # List trajectories
    trajectories = await file_storage.list_trajectories(project_id)
    assert len(trajectories) == 3
    assert "traj1" in trajectories
    assert "traj2" in trajectories
    assert "traj3" in trajectories


@pytest.mark.asyncio
async def test_list_evaluations(file_storage):
    """Test listing evaluations."""
    # Create test evaluation data
    eval1 = {"evaluation_name": "eval_test1", "status": "COMPLETED"}
    eval2 = {"evaluation_name": "eval_test2", "status": "RUNNING"}
    
    # Write evaluations directly to projects with the eval_ prefix
    await file_storage.write_to_project("evaluation", eval1, project_id="eval_test1")
    await file_storage.write_to_project("evaluation", eval2, project_id="eval_test2")
    
    # Debug: Check if project directory has been created properly
    base_dir = file_storage.base_dir
    print(f"Base directory: {base_dir}")
    print(f"Project 1 path: {base_dir / 'project' / 'eval_test1'}")
    print(f"Project 2 path: {base_dir / 'project' / 'eval_test2'}")
    
    # Debug: check what's in the list_projects result
    projects = await file_storage.list_projects()
    print(f"Projects: {projects}")
    
    # Debug: Try direct file inspection
    import glob
    print(f"All files in base directory: {glob.glob(str(base_dir / '**'), recursive=True)}")
    
    # List evaluations 
    evaluations = await file_storage.list_evaluations()
    print(f"Evaluations found: {evaluations}")
    
    assert len(evaluations) == 2
    
    eval_names = [e.get("evaluation_name") for e in evaluations]
    assert "eval_test1" in eval_names
    assert "eval_test2" in eval_names


@pytest.mark.asyncio
async def test_normalize_key():
    """Test the normalize_key method."""
    storage = FileStorage(base_dir=tempfile.mkdtemp())
    
    # Test various formats
    assert storage.normalize_key("key") == "key"
    assert storage.normalize_key("/key") == "key"
    assert storage.normalize_key("key/") == "key"
    assert storage.normalize_key("/key/") == "key"
    assert storage.normalize_key("  key  ") == "key"
    assert storage.normalize_key("/path/to/key/") == "path/to/key"


@pytest.mark.asyncio
async def test_error_handling(file_storage):
    """Test error cases."""
    # Reading non-existent key
    with pytest.raises(KeyError):
        await file_storage.read("nonexistent")
    
    # Accessing non-existent trajectory
    with pytest.raises(ValueError):
        await file_storage.read_from_trajectory("key", project_id="project", trajectory_id=None)


@pytest.mark.asyncio
async def test_concurrent_operations(file_storage):
    """Test concurrent operations on the same storage."""
    # Create multiple tasks that read/write
    async def write_task(key, value):
        await file_storage.write(key, {"value": value})
        return await file_storage.read(key)
    
    # Execute multiple write operations concurrently
    tasks = [write_task(f"concurrent_key_{i}", i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify results
    for i, result in enumerate(results):
        assert result == {"value": i}
    
    # Verify all keys exist
    for i in range(10):
        assert await file_storage.exists(f"concurrent_key_{i}") 