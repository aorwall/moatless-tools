"""
This is a customized version of the CLI in https://github.com/raymyers/swe-bench-util
"""
import csv
import json
import logging
import os
import subprocess
import sys
from typing import Optional, List

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

from benchmark.main import write_markdown, benchmark_retrieve
from benchmark.utils import diff_details
from moatless.retrievers.llama_index_retriever import IngestionPipelineSetup
from moatless.splitters.epic_split import CommentStrategy
from moatless.splitters.epic_split import EpicSplitter

app = typer.Typer()
get_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(get_app, name="get")
app.add_typer(run_app, name="run")


load_dotenv()

ingestion_pipelines = [
    #IngestionPipelineSetup(
    #    name="text-embedding-3-small--code-splitter-v2-1000",
    #    transformations=[
    #        CodeSplitterV2(chunk_size=1000, language="python")
    #    ],
    #    embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    #),
    IngestionPipelineSetup(
        name="text-embedding-3-small--epic-splitter-v3-100-750",
        splitter=EpicSplitter(chunk_size=750, min_chunk_size=100, language="python", comment_strategy=CommentStrategy.ASSOCIATE),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )
]


def maybe_clone(repo_url, repo_dir):
    if not os.path.exists(f"{repo_dir}/.git"):
        # Clone the repo if the directory doesn't exist
        result = subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, text=True, capture_output=True)

        if result.returncode == 0:
            print(f"Repo '{repo_url}' was cloned to '{repo_dir}'", file=sys.stderr)
        else:
            print(f"Failed to clone repo '{repo_url}' to '{repo_dir}'", file=sys.stderr)
            raise typer.Exit(code=1)
    else:
        print(f"Repo '{repo_url}' already exists in '{repo_dir}'", file=sys.stderr)


def checkout_commit(repo_dir, commit_hash):
    subprocess.run(['git', 'reset', '--hard', commit_hash], cwd=repo_dir, check=True)


def write_file(path, text):
    with open(path, 'w') as f:
        f.write(text)
        print(f"File '{path}' was saved", file=sys.stderr)


def write_json(path, name, data):
    json_str = json.dumps(data, indent=2)
    json_path = f"{path}/{name}.json"
    write_file(json_path, json_str)


def diff_file_names(text: str) -> list[str]:
    return [
        line[len("+++ b/"):] 
        for line in text.split('\n') 
        if line.startswith('+++')
    ]


def get_filename(split: str, dataset_name: str):
    return f'{dataset_name.replace("/", "-")}-{split}.json'


@get_app.command()
def download(split: str='dev', dataset_name='princeton-nlp/SWE-bench_Lite'):
    """Download oracle (patched files) for all rows in split"""

    file_name = get_filename(split, dataset_name)
    if os.path.exists(file_name):
        with open(file_name) as f:
            return json.load(f)

    dataset = load_dataset(dataset_name, split=split)
    result = []
    for row_data in dataset:
        row_data["patch_files"] = diff_file_names(row_data['patch'])
        row_data["test_patch_files"] = diff_file_names(row_data['test_patch'])
        row_data["patch_diff_details"] = diff_details(row_data['patch'])
        result.append(row_data)

    file_name = get_filename(split, dataset_name)
    write_json('data', file_name, result)

    return result


def get_case(id: str, split: str='dev', dataset_name='princeton-nlp/SWE-bench_Lite'):
    dataset = download(split, dataset_name)
    for row in dataset:
        if row['instance_id'] == id:
            return row


def calculate_precision_recall(recommended_files: List[str], patch_files: List[str]):
    true_positives = set(recommended_files) & set(patch_files)
    precision = len(true_positives) / len(recommended_files) if len(recommended_files) > 0 else 0
    recall = len(true_positives) / len(patch_files) if len(patch_files) > 0 else 0

    return precision, recall


@run_app.command()
def suite(suite: str = 'retries'):
    suites = json.load(open('benchmark/suites.json'))
    for instance_id in suites[suite]:
        instance(instance_id, suite)


@run_app.command()
def dataset(split: str='dev', dataset_name='princeton-nlp/SWE-bench_Lite'):
    ds = download(split, dataset_name)
    ds = sorted(ds, key=lambda x: x['created_at'])

    benchmark_run = f"{dataset_name.replace('/', '-')}-{split}"

    if not os.path.exists(f"benchmark/reports/{benchmark_run}"):
        os.makedirs(f"benchmark/reports/{benchmark_run}")

    with open(f"benchmark/reports/{benchmark_run}/summary.csv", "w") as f:
        csv.writer(f, delimiter=";").writerow(["instance_id", "pipeline", "vectors", "tokens", "no_of_patches", "context_length", "avg_pos", "min_pos", "max_pos", "top_file_pos", "missing_snippets", "missing_patch_files"])

    for i, row in enumerate(ds):
        print(f"{i + 1}/{len(ds)}: {row['instance_id']}")
        run_benchmark(row, benchmark_run)


@run_app.command()
def instance(
        id: str,
        benchmark_run: str = typer.Option(default='single_run')):
    print(f"Run benchmark on {id}")
    row_data = get_case(id=id)
    run_benchmark(row_data, benchmark_run)


def run_benchmark(row_data: dict, benchmark_run: str):
    repo_name = row_data['repo'].split('/')[-1]
    repo = f'git@github.com:{row_data["repo"]}.git'
    base_commit = row_data['base_commit']
    path = f'/tmp/repos/{repo_name}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' was created.")
    maybe_clone(repo, path)
    checkout_commit(path, base_commit)

    instance_id = row_data['instance_id']
    print(f"Processing {instance_id}")
    if not os.path.exists(f"benchmark/reports/{benchmark_run}/{instance_id}"):
        os.makedirs(f"benchmark/reports/{benchmark_run}/{instance_id}")

    for pipeline_setup in ingestion_pipelines:
        benchmark_retrieve(pipeline_setup=pipeline_setup, benchmark_run=benchmark_run, path=path, repo_name=repo_name, row_data=row_data, commit=base_commit)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        is_eager=True,
    )
) -> None:
    return



if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    instance('pydicom__pydicom-1256', benchmark_run='debug')