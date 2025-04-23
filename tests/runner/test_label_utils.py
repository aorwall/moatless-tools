import pytest
from moatless.runner.label_utils import create_resource_id


def test_create_resource_id_basic():
    """Test basic functionality of create_resource_id."""
    resource_id = create_resource_id("test-project", "test-trajectory", "run")
    # New format with hash
    assert resource_id.startswith("run-test-project-test-trajectory-")
    # Hash should be 8 characters
    assert len(resource_id.split("-")[-1]) == 8

    # With different prefix
    resource_id = create_resource_id("test-project", "test-trajectory", "moatless")
    assert resource_id.startswith("moatless-test-project-test-trajectory-")
    assert len(resource_id.split("-")[-1]) == 8


def test_create_resource_id_special_characters():
    """Test special character handling in create_resource_id."""
    resource_id = create_resource_id("test_project/123", "test.trajectory@example", "run")
    print(f"Special chars resource_id: {resource_id}")

    # Format: prefix-proj_prefix-traj_suffix-hash
    parts = resource_id.split("-")
    assert parts[0] == "run"

    # Check that the ID contains expected substrings
    assert "test" in resource_id
    assert "project" in resource_id or "123" in resource_id
    assert "trajectory" in resource_id
    assert "example" in resource_id

    # Hash should be 8 characters
    assert len(parts[-1]) == 8

    # With invalid start/end characters
    resource_id = create_resource_id("-test-", "-trajectory-", "run")
    assert resource_id.startswith("run-")
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

    # Check that both IDs are represented - project should be in the first 16 chars
    assert "very-long-projec" in resource_id
    assert "trajectory" in resource_id

    # Check format - prefix + hash at the end
    assert resource_id.startswith("run-")
    assert len(resource_id.split("-")[-1]) == 8


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
    assert "proje-t-" in resource_id
    # Hash at the end
    assert len(resource_id.split("-")[-1]) == 8


def test_resource_id_consistency():
    """Test that the same inputs always produce the same ID."""
    id1 = create_resource_id("project-x", "trajectory-y", "run")
    id2 = create_resource_id("project-x", "trajectory-y", "run")

    assert id1 == id2

    # Different inputs should produce different IDs
    id3 = create_resource_id("project-x", "trajectory-z", "run")
    assert id1 != id3


def test_resource_id_hash_uniqueness():
    """Test that different inputs produce different hash values."""
    id1 = create_resource_id("project-a", "trajectory-1", "run")
    id2 = create_resource_id("project-b", "trajectory-1", "run")
    id3 = create_resource_id("project-a", "trajectory-2", "run")

    # Extract hash parts
    hash1 = id1.split("-")[-1]
    hash2 = id2.split("-")[-1]
    hash3 = id3.split("-")[-1]

    # Verify hashes are different for different inputs
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash2 != hash3
