import json
import logging
import os
import shutil
import tempfile
from typing import Optional

from datasets import load_dataset

from moatless.index.code_index import CodeIndex
from moatless.index.settings import IndexSettings
from moatless.utils.repo import setup_github_repo, get_repo_dir_name

logger = logging.getLogger(__name__)


def create_index(
    repo_path: str,
    benchmark_name: str,
    index_settings: IndexSettings,
    instance_id: str,
    persist_dir: str,
):
    try:
        code_index = CodeIndex.from_persist_dir(persist_dir)
    except Exception:
        logger.info("Create new index")
        code_index = CodeIndex(settings=index_settings)

    vectors, indexed_tokens = code_index.run_ingestion(repo_path=repo_path)
    logger.info(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")

    code_index.persist(persist_dir=persist_dir)

    try:
        upload_store(persist_dir, benchmark_name, instance_id)
    except Exception as e:
        logger.info(f"Failed to upload store: {e}")

    return code_index


def upload_store(persist_dir: str, benchmark_name: str, instance_id: str):
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        logger.info("AZURE_STORAGE_CONNECTION_STRING is not set, cannot upload store.")
        return

    try:
        from azure.storage.blob import BlobServiceClient
    except:
        logger.info(
            "Azure Storage Blobs client not installed, cannot upload store. Install with 'pip install azure-storage-blob'"
        )
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zip_file = os.path.join(temp_dir, instance_id)
        shutil.make_archive(temp_zip_file, "zip", persist_dir)

        block_storage_client = BlobServiceClient.from_connection_string(connect_str)
        _blob_storage = block_storage_client.get_container_client(container="stores")
        blob_name = f"{benchmark_name}/{instance_id}.zip"

        blob_client = _blob_storage.get_blob_client(blob=blob_name)
        with open(temp_zip_file, "rb") as data:
            blob_client.upload_blob(data)

        logger.info(f"Uploaded {blob_name} to Azure Blob Storage.")


def create_and_benchmark_index(
    benchmark_name: str,
    settings: IndexSettings,
    evaluations_dir: str = "evaluations/code_index",
    repo_dir: str = "/tmp/repos",
    index_perist_dir: Optional[str] = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    instance_ids: Optional[list] = None,
):
    instances = load_dataset(dataset_name, split=split)
    instances = sorted(instances, key=lambda x: x["created_at"])

    existing_instance_ids = set()

    if os.path.exists(f"{evaluations_dir}/{benchmark_name}.jsonl"):
        with open(f"{evaluations_dir}/{benchmark_name}.jsonl", "r") as file:
            report_files = file.readlines()
            for line in report_files:
                data = json.loads(line)
                existing_instance_ids.add(data["instance_id"])

    for i, instance_data in enumerate(instances):
        if instance_ids and instance_data["instance_id"] not in instance_ids:
            continue
        elif instance_data["instance_id"] in existing_instance_ids:
            logger.info(
                f"Skipping existing instance {instance_data['instance_id']} ({i}/ {len(instances)})"
            )
            continue

        logger.info(
            f"Processing instance {instance_data['instance_id']} ({i}/ {len(instances)})"
        )

        repo_path = setup_github_repo(
            repo=instance_data["repo"],
            base_commit=instance_data["base_commit"],
            base_dir=repo_dir,
        )

        repo_index_dir = os.path.join(
            index_perist_dir, get_repo_dir_name(instance_data["repo"])
        )

        index = create_index(
            repo_path=repo_path,
            benchmark_name=benchmark_name,
            index_settings=settings,
            persist_dir=repo_index_dir,
            instance_id=instance_data["instance_id"],
        )

        result = index.finish(instance_data["problem_statement"])
        print(result)
