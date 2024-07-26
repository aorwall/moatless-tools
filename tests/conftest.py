# my_project/tests/conftest.py
import os
from dotenv import load_dotenv
import pytest

# Load environment variables from .env file if it exists
load_dotenv()

def pytest_addoption(parser):
    parser.addoption("--index-store-dir", action="store", default=os.getenv('INDEX_STORE_DIR', '/tmp/index-store'), help="Path for INDEX_STORE_DIR")
    parser.addoption("--repo-dir", action="store", default=os.getenv('REPO_DIR', '/tmp/repos'), help="Path for REPO_DIR")

@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch, request):
    index_store_dir = request.config.getoption("--index-store-dir")
    repo_dir = request.config.getoption("--repo-dir")

    monkeypatch.setenv('INDEX_STORE_DIR', index_store_dir)
    monkeypatch.setenv('REPO_DIR', repo_dir)
