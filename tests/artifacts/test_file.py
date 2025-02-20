import os
from pathlib import Path
import pytest
from moatless.artifacts.file import FileArtifactHandler, TextFileArtifact

@pytest.fixture
def test_directory(tmp_path):
    return tmp_path / "test_artifacts"

@pytest.fixture
def file_handler(test_directory):
    return FileArtifactHandler(directory_path=test_directory)

@pytest.fixture
def sample_pdf_path():
    # Get the current test file's directory
    current_dir = Path(__file__).parent
    return current_dir / "data" / "dummy.pdf"

def test_pdf_reading(file_handler, sample_pdf_path, test_directory):
    # Create test directory if it doesn't exist
    test_directory.mkdir(parents=True, exist_ok=True)
    
    # Copy the PDF to the test directory
    test_pdf_path = test_directory / "dummy.pdf"
    test_pdf_path.write_bytes(sample_pdf_path.read_bytes())
    
    # Load and test the PDF artifact
    artifact = file_handler.load("dummy.pdf")
    
    assert isinstance(artifact, TextFileArtifact)
    assert artifact.mime_type == "application/pdf"
    assert "Dummy PDF file" in artifact.content 