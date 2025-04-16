import pytest
from moatless.runner.label_utils import create_resource_id


def test_create_resource_id_basic():
    """Test basic functionality of create_resource_id."""
    resource_id = create_resource_id("test-project", "test-trajectory", "run")
    assert resource_id == "run-test-project-test-trajectory"
    
    # With different prefix
    resource_id = create_resource_id("test-project", "test-trajectory", "moatless")
    assert resource_id == "moatless-test-project-test-trajectory"


def test_create_resource_id_special_characters():
    """Test special character handling in create_resource_id."""
    resource_id = create_resource_id("test_project/123", "test.trajectory@example", "run")
    assert resource_id == "run-test-project-123-test-trajectory-example"
    
    # With invalid start/end characters
    resource_id = create_resource_id("-test-", "-trajectory-", "run")
    assert resource_id.startswith("run-x")
    assert not resource_id.startswith("run--")
    assert not resource_id.endswith("-")


def test_create_resource_id_length_limits():
    """Test length limitations in create_resource_id."""
    # Create very long project and trajectory IDs
    long_project_id = "very-long-project-id-" + "x" * 50
    long_trajectory_id = "very-long-trajectory-id-" + "y" * 50
    
    resource_id = create_resource_id(long_project_id, long_trajectory_id, "run")
    
    # Check length constraint
    assert len(resource_id) <= 63
    
    # Check that both IDs are represented in some form
    assert "project" in resource_id
    assert "trajectory" in resource_id
    
    # Check format 
    assert resource_id.startswith("run-")


def test_create_resource_id_validation():
    """Test that empty IDs raise ValueError."""
    # Test with empty project_id
    with pytest.raises(ValueError, match="project_id must be provided and not empty"):
        create_resource_id("", "trajectory", "run")
    
    # Test with empty trajectory_id
    with pytest.raises(ValueError, match="trajectory_id must be provided and not empty"):
        create_resource_id("project", "", "run")
        
    # Test with both empty
    with pytest.raises(ValueError):
        create_resource_id("", "", "run")


def test_create_resource_id_with_unicode():
    """Test handling of unicode characters in create_resource_id."""
    resource_id = create_resource_id("projeçt-ünicode", "trajéctory-chärs", "run")
    # Unicode characters should be replaced with dashes
    assert "ç" not in resource_id
    assert "ü" not in resource_id
    assert "é" not in resource_id
    assert "ä" not in resource_id
    assert resource_id.startswith("run-proje-t-")


def test_resource_id_consistency():
    """Test that the same inputs always produce the same ID."""
    id1 = create_resource_id("project-x", "trajectory-y", "run")
    id2 = create_resource_id("project-x", "trajectory-y", "run")
    
    assert id1 == id2
    
    # Different inputs should produce different IDs
    id3 = create_resource_id("project-x", "trajectory-z", "run")
    assert id1 != id3 