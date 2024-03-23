"""
This is a customized version of the CLI in https://github.com/raymyers/swe-bench-util
"""
import json
import os
import subprocess
import sys
from typing import Optional, List

import typer
from datasets import load_dataset
from llama_index.embeddings.openai import OpenAIEmbedding

from moatless.retrievers.llama_index_retriever import IngestionPipelineSetup, LlamaIndexCodeSnippetRetriever
from moatless.splitters.epic_split import EpicSplitter
from moatless.splitters.epic_split import CommentStrategy

app = typer.Typer()
get_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(get_app, name="get")
app.add_typer(run_app, name="run")

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
        transformations=[
            EpicSplitter(chunk_size=750, min_chunk_size=100, language="python", comment_strategy=CommentStrategy.ASSOCIATE)
        ],
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

def format_markdown_code_block(text):
    text = str(text).replace('```', '\\`\\`\\`')
    return f"```\n{text}\n```"

def write_markdown(path, name, data):
    text = f"""# {data['instance_id']}

* repo: {data['repo']}
* base_commit: `{data['base_commit']}`

## Problem statement

{data['problem_statement']}

## Patch

```diff
{data['patch']}
```

## Test Patch

```diff
{data['test_patch']}
```
"""

    md_path = f"{path}/{name}.md"
    write_file(md_path, text)


def diff_file_names(text: str) -> list[str]:
    return [
        line[len("+++ b/"):] 
        for line in text.split('\n') 
        if line.startswith('+++')
    ]


@get_app.command()
def download(split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Download oracle (patched files) for all rows in split"""
    dataset = load_dataset(dataset_name, split=split)
    result = []
    for row_data in dataset:
        row_data["patch_files"] = diff_file_names(row_data['patch'])
        row_data["test_patch_files"] = diff_file_names(row_data['test_patch'])
        row_data["id"] = diff_file_names(row_data['instance_id'])
        result.append(row_data)
    write_json('rows', "oracle", result)


def get_case(id: str):
    with open(f'data/oracle.json') as f:
        oracle_json = json.load(f)

    for row in oracle_json:
        if row['instance_id'] == id:
            return row


def calculate_precision_recall(recommended_files: List[str], patch_files: List[str]):
    true_positives = set(recommended_files) & set(patch_files)
    precision = len(true_positives) / len(recommended_files) if len(recommended_files) > 0 else 0
    recall = len(true_positives) / len(patch_files) if len(patch_files) > 0 else 0

    return precision, recall


def benchmark_retrieve(pipeline_setup: IngestionPipelineSetup, path: str, repo_name: str, case: dict):
    retriever = LlamaIndexCodeSnippetRetriever.from_pipeline_setup(
        pipeline_setup=pipeline_setup,
        path=path,
        perist_dir=f"/tmp/repos/{repo_name}-storage/{pipeline_setup.name}",
    )

    vectors, tokens = retriever.run_index()

    patch_files = [f"{path}/{file}" for file in case["patch_files"]]
    case_id = case["instance_id"]

    query = case["problem_statement"]

    result = retriever.retrieve(query)

    no_of_patches = len(patch_files)

    found_patch_files = set()
    recommended_files = []

    position = 0
    min_pos = None
    max_pos = None
    sum_pos = 0

    md = ""
    for i, snippet in enumerate(result):
        if snippet.path in patch_files:
            found_patch_files.add(snippet.path)
            md += f"**{i} - {snippet.path}**:\n```python\n{snippet.content}\n```\n"
        else:
            md += f"{i} - {snippet.path}\n```python\n{snippet.content}\n```\n"

        if snippet.path not in recommended_files:
            recommended_files.append(snippet.path)
            position += 1

            if snippet.path in patch_files:
                print(f" -> {position}: {snippet.path}")
                if min_pos is None:
                    min_pos = position
                max_pos = position
                sum_pos += position
            else:
                print(f"    {position}: {snippet.path}")

        if len(found_patch_files) == len(patch_files):
            break

    with open(f"reports/{case_id}/{pipeline_setup.name}.md", "w") as f:
        f.write(md)

    precision, recall = calculate_precision_recall(recommended_files, patch_files)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    avg_pos = sum_pos / no_of_patches if no_of_patches > 0 else 0
    missing_patch_files = len(patch_files) - len(found_patch_files)

    with open(f"reports/summary.csv", "a") as f:
        f.write(f"{case_id}:{pipeline_setup.name}:{vectors}:{tokens}:{precision}:{recall}:{no_of_patches}:{avg_pos}:{min_pos}:{max_pos}:{missing_patch_files}\n")

    print("\nMissing Patch Files:")
    for i, file_path in enumerate(patch_files, start=1):
        print(f"{i}: {file_path}")


@run_app.command()
def suite(suite: str = 'retries'):
    suites = json.load(open('benchmark/suites.json'))
    for case_id in suites[suite]:
        case(case_id)


@run_app.command()
def case(case_id: str):
    row_data = get_case(id=case_id)
    repo_name = row_data['repo'].split('/')[-1]
    repo = f'git@github.com:{row_data["repo"]}.git'
    base_commit = row_data['base_commit']
    path = f'/tmp/repos/{repo_name}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' was created.")
    maybe_clone(repo, path)
    checkout_commit(path, base_commit)

    print(f"Processing {case_id}")
    if not os.path.exists(f"reports/{case_id}"):
        os.makedirs(f"reports/{case_id}")

    write_markdown(f'reports/{case_id}', "issue", row_data)

    for pipeline_setup in ingestion_pipelines:
        benchmark_retrieve(pipeline_setup, path, repo_name, row_data)

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

