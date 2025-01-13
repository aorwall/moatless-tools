import logging
import os

import pytest
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TEST_MODELS = [
    "claude-3-5-sonnet-20240620",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "deepseek/deepseek-chat",
]


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


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch, request):
    index_store_dir = request.config.getoption("--index-store-dir")
    repo_dir = request.config.getoption("--repo-dir")

    logger.debug(f"Setting INDEX_STORE_DIR={index_store_dir}")
    logger.debug(f"Setting REPO_DIR={repo_dir}")

    if not os.path.exists(index_store_dir):
        os.makedirs(index_store_dir)

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)

    monkeypatch.setenv("INDEX_STORE_DIR", index_store_dir)
    monkeypatch.setenv("REPO_DIR", repo_dir)


# Add the llm_integration marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm_integration: mark test as requiring LLM integration"
    )
