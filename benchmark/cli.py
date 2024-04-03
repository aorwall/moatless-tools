"""
This is a customized version of the CLI in https://github.com/raymyers/swe-bench-util
"""
import json
import logging
import os
import sys
from typing import Optional, List

import typer
from datasets import load_dataset
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from moatless.retrievers.golden_retriever import IngestionPipelineSetup

from benchmark.main import benchmark_retrieve
from benchmark.swebench import download
from benchmark.utils import diff_details
from moatless.splitters.epic_split import CommentStrategy
from moatless.splitters.epic_split import EpicSplitter
from moatless.utils import setup_github_repo

app = typer.Typer()
get_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(get_app, name="get")
app.add_typer(run_app, name="run")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

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
        name="epic-splitter-v4--100-750--comment-associate--text-embedding-3-small--1536",
        splitter=EpicSplitter(min_chunk_size=100, chunk_size=750, language="python", comment_strategy=CommentStrategy.ASSOCIATE),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )
]


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
    return f'{dataset_name.replace("/", "-")}-{split}'



def get_case(id: str, split: str='test', dataset_name='princeton-nlp/SWE-bench_Lite'):
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
def dataset(split: str='dev', dataset_name='princeton-nlp/SWE-bench_Lite', report_dir='benchmark/reports'):
    ds = download(split, dataset_name)

    print(f"Sorted and start with {ds[0]['instance_id']}")

    benchmark_run = f"{dataset_name.replace('/', '-')}-{split}"

    if not os.path.exists(f"{report_dir}/{benchmark_run}"):
        os.makedirs(f"{report_dir}/{benchmark_run}")

    benchmarked_instances = set()
    if os.path.exists(f"{report_dir}/{benchmark_run}/summary.json"):
        with open(f"{report_dir}/{benchmark_run}/summary.json") as f:
            summary = json.load(f)
            for row in summary:
                benchmarked_instances.add(row['instance_id'])

    benchmarks = [row for row in ds if row['instance_id'] not in benchmarked_instances]
    benchmarks.sort(key=sort_key)

    for i, row in enumerate(benchmarks):
        print(f"{i + 1}/{len(benchmarks)}: {row['instance_id']}")
        run_benchmark(row, benchmark_run, report_dir=report_dir)

def sort_key(data_row):
    text, number = data_row["instance_id"].rsplit('-', 1)
    return text, int(number)

@run_app.command()
def instance(
        id: str,
        benchmark_run: str = typer.Option(default='single_run'),
        split: str = typer.Option(default='test'),
        dataset_name: str = typer.Option(default='princeton-nlp/SWE-bench_Lite')):
    print(f"Run benchmark on {id}")
    row_data = get_case(id=id, split=split, dataset_name=dataset_name)
    run_benchmark(row_data, benchmark_run)




def run_benchmark(row_data: dict, benchmark_run: str, report_dir='benchmark/reports'):
    setup_github_repo(row_data["repo"], row_data['base_commit'])

    instance_id = row_data['instance_id']
    print(f"Processing {instance_id}")
    if not os.path.exists(f"{report_dir}/{benchmark_run}/{instance_id}"):
        os.makedirs(f"{report_dir}/{benchmark_run}/{instance_id}")

    for pipeline_setup in ingestion_pipelines:
        benchmark_retrieve(pipeline_setup=pipeline_setup, benchmark_run=benchmark_run, path=path, repo_name=repo_name, row_data=row_data, commit=base_commit, report_dir=report_dir)


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
    dataset(split='devin', dataset_name='princeton-nlp-SWE-bench', report_dir='reports')