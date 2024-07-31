import logging
import os
from dotenv import load_dotenv
import pytest

load_dotenv()
logger = logging.getLogger(__name__)


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
        "--moatless-dir",
        action="store",
        default=os.getenv("MOATLESS_DIR", "/tmp/moatless"),
        help="Path for MOATLESS_DIR",
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
    moatless_dir = request.config.getoption("--moatless-dir")

    logger.debug(f"Setting INDEX_STORE_DIR={index_store_dir}")
    logger.debug(f"Setting REPO_DIR={repo_dir}")
    logger.debug(f"Setting MOATLESS_DIR={moatless_dir}")

    if not os.path.exists(index_store_dir):
        os.makedirs(index_store_dir)

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)

    if not os.path.exists(moatless_dir):
        os.makedirs(moatless_dir)

    monkeypatch.setenv("INDEX_STORE_DIR", index_store_dir)
    monkeypatch.setenv("REPO_DIR", repo_dir)
    monkeypatch.setenv("MOATLESS_DIR", moatless_dir)
