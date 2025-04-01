import logging
import os
import shutil
import tempfile

import pytest
from dotenv import load_dotenv

# Import here to avoid circular imports
from moatless import settings
from moatless.storage.base import BaseStorage
from moatless.storage.file_storage import FileStorage

load_dotenv()
logger = logging.getLogger(__name__)

TEST_MODELS = [
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "deepseek/deepseek-chat",
]


@pytest.fixture
def temp_dir():
    """Fixture that creates a temporary directory for testing."""
    test_dir = tempfile.mkdtemp(prefix="moatless_test_")
    logger.debug(f"Created temporary test directory: {test_dir}")
    yield test_dir
    # Clean up after test
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def file_storage(temp_dir):
    """Fixture to create a storage instance with a test directory."""
    
    settings._storage = FileStorage(base_dir=temp_dir)
    return settings._storage


def pytest_addoption(parser):
    parser.addoption(
        "--index-store-dir",
        action="store",
        default=os.getenv("INDEX_STORE_DIR", "/tmp/index-store"),
        help="Path for INDEX_STORE_DIR",
    )
    parser.addoption(
        "--repo-dir",
        action="store",
        default=os.getenv("REPO_DIR", "/tmp/repo"),
        help="Path for REPO_DIR",
    )
    parser.addoption(
        "--run-llm-integration",
        action="store_true",
        default=False,
        help="run integration tests that call LLMs",
    )
    parser.addoption(
        "--run-with-index",
        action="store_true",
        default=False,
        help="run tests that need vector store index files",
    )
    parser.addoption(
        "--moatless-dir",
        action="store",
        default=None,
        help="Set a specific MOATLESS_DIR for testing; defaults to temporary directory",
    )


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch, request, temp_dir):
    index_store_dir = request.config.getoption("--index-store-dir")
    repo_dir = request.config.getoption("--repo-dir")
    moatless_dir = request.config.getoption("--moatless-dir")

    logger.debug(f"Setting INDEX_STORE_DIR={index_store_dir}")
    logger.debug(f"Setting REPO_DIR={repo_dir}")
    logger.debug(f"Setting MOATLESS_DIR={temp_dir}")

    if not os.path.exists(index_store_dir):
        os.makedirs(index_store_dir)

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
        
    # Create MOATLESS_DIR if specified on command line, otherwise tempfile is used
    if moatless_dir:
        logger.debug(f"Setting MOATLESS_DIR={moatless_dir}")
        if not os.path.exists(moatless_dir):
            os.makedirs(moatless_dir)
        monkeypatch.setenv("MOATLESS_DIR", moatless_dir)

    monkeypatch.setenv("INDEX_STORE_DIR", index_store_dir)
    monkeypatch.setenv("REPO_DIR", repo_dir)

    yield
    
    # Clean up temp directories after tests
    if moatless_dir and os.path.exists(moatless_dir) and "--keep-files" not in request.config.getoption:
        shutil.rmtree(moatless_dir, ignore_errors=True)


# Add the llm_integration marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm_integration: mark test as requiring LLM integration"
    )
