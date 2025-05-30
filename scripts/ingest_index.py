import argparse
import json
import logging
import os
import shutil
from typing import Optional, Dict, List

from dotenv import load_dotenv
from moatless.benchmark.swebench import (
    get_repo_dir_name,
)
from moatless.benchmark.swebench.utils import create_repository
from moatless.evaluation.utils import get_moatless_instances
from moatless.index.settings import IndexSettings
from moatless.index.code_index import CodeIndex
from moatless.repository.git import GitRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_repo(repo_dir: str, index_settings: IndexSettings) -> None:
    repository = GitRepository.from_path(repo_dir)
    code_index = CodeIndex(file_repo=repository, settings=index_settings)
    vectors, indexed_tokens = code_index.run_ingestion(repo_path=repo_dir, num_workers=1)
    print(f"Indexed {indexed_tokens} tokens and created {vectors} vectors")
    persist_dir = os.path.join(repo_dir, ".moatless", "index")
    code_index.persist(persist_dir=persist_dir)


def main():
    parser = argparse.ArgumentParser(description="Ingest and create code index for SWE-Bench instances")
    parser.add_argument("--repo-dir", required=True, help="Directory code base to ingest")
    parser.add_argument("--index-store-dir", required=False, help="Directory to store vector index file")
    parser.add_argument("--embed-model", default="voyage-code-3", help="Embedding model to use")
    
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Initialize index settings
    index_settings = IndexSettings(embed_model=args.embed_model, dimensions=1024, language="java")

    ingest_repo(args.repo_dir, index_settings)

if __name__ == "__main__":
    main()
