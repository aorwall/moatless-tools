import csv
import json
import logging
import os
from typing import Optional

import litellm
import typer
import subprocess
from dotenv import load_dotenv
from llama_index.embeddings.voyageai import VoyageEmbedding

from benchmark import swebench
from benchmark.utils import download_or_create_index
from moatless.codeblocks.utils import Colors
from moatless.search import Search
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy
from moatless.utils.repo import setup_github_repo


load_dotenv("../.env")

base_dir = "/tmp/repos"

app = typer.Typer()

filtered = []


def run_instance(
    instance_data: dict, persist_dir: str = "/tmp/storage", model: str = "gpt-4-turbo"
):
    print(
        f"{Colors.YELLOW}Running instance: {instance_data['instance_id']}{Colors.RESET}"
    )

    csv_path = f"found_files_{model}.csv"
    print(f"CSV path: {csv_path}")

    if os.path.exists(csv_path):
        with open(csv_path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == instance_data["instance_id"] and not row[0] in filtered:
                    print(f"Skipping instance {instance_data['instance_id']}")
                    return row[1] == "True"

    repo_path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )
    print(f"{Colors.YELLOW}Cloned repo to path: {repo_path}{Colors.RESET}")

    repo_dir_name = instance_data["repo"].replace("/", "__")

    persist_dir = f"{persist_dir}/{repo_dir_name}"

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    embed_model = VoyageEmbedding(
        model_name="voyage-code-2",
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        truncation=False,
        embed_batch_size=60,
    )

    splitter = EpicSplitter(
        min_chunk_size=100,
        chunk_size=1500,
        hard_token_limit=2000,
        max_chunks=200,
        language="python",
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    if instance_data["instance_id"].startswith(
        "sympy"
    ):  # FIXME: Workaround because the sympy vectors ended up in another dir
        ingestion_name = "voyage-code-2-100-1000"
    else:
        ingestion_name = "voyage-code-2-100-1500"

    try:
        code_index = download_or_create_index(
            persist_dir=persist_dir,
            ingestion_name=ingestion_name,
            repo_path=repo_path,
            repo_name=instance_data["repo"],
            base_commit=instance_data["base_commit"],
            embed_model=embed_model,
            splitter=splitter,
        )

        if not code_index:
            print(f"{Colors.RED}Failed to create retriever{Colors.RESET}")
            return None

        log_dir = f"logs/search_{model}/{instance_data['instance_id']}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return

        search = Search(
            code_index=code_index,
            path=repo_path,
            model=model,
            log_dir=f"logs/search/{instance_data['instance_id']}",
            # metadata={"tags": [ingestion_name, instance_data["instance_id"]]},
        )

        file_path = search.search(instance_data["problem_statement"])
        patch_file = instance_data["patch_files"][0]

        with open(csv_path, "a") as file:
            print(f"Writing to {csv_path}")
            csv_writer = csv.writer(file)
            csv_writer.writerow(
                [
                    instance_data["instance_id"],
                    patch_file == file_path,
                    file_path,
                    patch_file,
                ]
            )

        if patch_file == file_path:
            print(
                f"Expected {Colors.YELLOW}{patch_file}{Colors.RESET}, got {Colors.GREEN}{file_path}{Colors.RESET},"
            )
        else:
            print(
                f"Expected {Colors.YELLOW}{patch_file}{Colors.RESET}, got {Colors.RED}{file_path}{Colors.RESET},"
            )

        return patch_file == file_path

    except Exception as e:
        print(
            f"{Colors.RED}Failed to run instance: {instance_data['instance_id']}{Colors.RESET}"
        )

        import traceback

        traceback.print_exc()

        with open("failed.txt", "a") as file:
            file.write(f"{instance_data['instance_id']}\n")

        return None


def _get_run_instance_ids(file_path):
    instance_ids = set()
    if not os.path.exists(file_path):
        return instance_ids

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            instance_ids.add(data["instance_id"])
    return instance_ids


def run_single_instance(instance_id):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )
    run_instance(instance_data)


def run_instances(
    split: str, dataset_name: str, data_dir: str, model: str = "gpt-4-turbo"
):
    instances = swebench.get_instances(
        split=split, dataset_name=dataset_name, data_dir=data_dir
    )

    total = 0
    success = 0
    for instance_data in instances:
        result = run_instance(instance_data, model=model)
        if result is not None:
            total += 1

        if result:
            success += 1

        if total:
            print(
                f"Benchmark run {total} / {len(instances)}, success rate: {success/total}"
            )


@app.command()
def benchmark(
    split="test",
    dataset_name="princeton-nlp/SWE-bench_Lite",
    data_dir="../data",
    model: str = "gpt-4-turbo",
):
    run_instances(split, dataset_name, data_dir, model)


if "__main__" == __name__:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.INFO)
    run_instances(
        split="test",
        dataset_name="princeton-nlp/SWE-bench_Lite",
        data_dir="../data",
        model="gpt-4-turbo",  # "claude-3-haiku-20240307"
    )
