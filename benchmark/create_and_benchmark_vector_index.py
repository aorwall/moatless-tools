import logging
import os
import shutil

from azure.storage.blob import BlobServiceClient
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding

from benchmark.main import upload_store
from benchmark.swebench import get_instances
from moatless.ingestion import CodeBaseIngestionPipeline
from moatless.splitters.epic_split import CommentStrategy, EpicSplitter
from moatless.utils.repo import setup_github_repo

logger = logging.getLogger(__name__)


def create_index(
    path: str,
    index_name: str,
    embed_model: BaseEmbedding,
    chunk_size: int = 1000,
    min_chunk_size: int = 100,
    persist_dir: str = ".storage",
):
    ingestion = CodeBaseIngestionPipeline.from_path(
        path=path,
        perist_dir=persist_dir,
        embed_model=embed_model,
        splitter=EpicSplitter(
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            language="python",
            comment_strategy=CommentStrategy.ASSOCIATE,
        ),
    )

    vectors, indexed_tokens = ingestion.run()
    print(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")

    ingestion.persist(persist_dir=persist_dir)

    ingestion_name = f"{embed_model.model_name}-{min_chunk_size}-{chunk_size}"
    try:
        upload_store(persist_dir, ingestion_name, index_name)
    except Exception as e:
        logger.info(f"Failed to upload store: {e}")

    return ingestion.retriever()


def upload_store(store_path: str, ingestion_name: str, file_name: str):
    shutil.make_archive(file_name, "zip", store_path)

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        logger.info("AZURE_STORAGE_CONNECTION_STRING is not set, cannot upload store.")
        return

    block_storage_client = BlobServiceClient.from_connection_string(connect_str)
    _blob_storage = block_storage_client.get_container_client(container="stores")
    blob_name = f"{ingestion_name}/{file_name}.zip"

    blob_client = _blob_storage.get_blob_client(blob=blob_name)
    with open(f"{file_name}.zip", "rb") as data:
        blob_client.upload_blob(data)

    logger.info(f"Uploaded {blob_name} to Azure Blob Storage.")


def run_instances(
    name: str,
    model: BaseEmbedding,
    min_chunk_size: int,
    chunk_size: int,
    repo_dir: str = "/tmp/repos",
):
    instances = get_instances(
        split="test", dataset_name="princeton-nlp/SWE-bench_Lite", data_dir="../data"
    )

    if not os.path.exists(f"reports/{name}.csv"):
        with open(f"reports/{name}.csv", "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(
                [
                    "instance_id",
                    "patch_file",
                    "file_tokens",
                    "selected_tokens",
                    "token_recall",
                    "spans",
                    "snippets",
                ]
            )

    existing_instance_ids = set()
    if os.path.exists(f"reports/{name}.jsonl"):
        with open(f"reports/{name}.jsonl", "r") as file:
            report_files = file.readlines()
            for line in report_files:
                data = json.loads(line)
                existing_instance_ids.add(data["instance_id"])

    for i, instance_data in enumerate(instances):
        print(
            f"Processing instance {instance_data['instance_id']} ({i}/ {len(instances)})"
        )
        if instance_data["instance_id"] in existing_instance_ids:
            print(f"Skipping instance {instance_data['instance_id']}")
            continue

        repo_path = setup_github_repo(
            repo=instance_data["repo"],
            base_commit=instance_data["base_commit"],
            base_dir=repo_dir,
        )

        retriever = create_index(
            path=instance_data["repo_path"],
            index_name=name,
            embed_model=model,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
        )

        result = find_snippet_vectors(instance_data, model, min_chunk_size, chunk_size)

        with open(f"reports/{name}.jsonl", "a") as file:
            json_string = json.dumps(result)
            file.write(json_string + "\n")

        token_recall = result["selected_tokens"] / result["file_tokens"]
        token_recall = int(token_recall * 100)

        with open(f"reports/{name}.csv", "a") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(
                [
                    instance_data["instance_id"],
                    result["patch_file"],
                    result["file_tokens"],
                    result["selected_tokens"],
                    token_recall,
                    len(result["spans"]),
                    result["snippets"],
                ]
            )


if __name__ == "__main__":
    voyage_embedding = VoyageEmbedding(
        model_name="voyage-code-2",
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        truncation=False,
    )

    run_instances("swe-bench-lite-voyage-code-2", voyage_embedding, 100, 1500)
