import csv
import os
import shutil
import sys
from typing import List

import chromadb
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from llama_index.core import get_tokenizer
from llama_index.vector_stores.chroma import ChromaVectorStore

from moatless.retrievers.llama_index_retriever import IngestionPipelineSetup, LlamaIndexCodeSnippetRetriever
from moatless.splitters import report


def calculate_precision_recall(recommended_files: List[str], patch_files: List[str]):
    true_positives = set(recommended_files) & set(patch_files)
    precision = len(true_positives) / len(recommended_files) if len(recommended_files) > 0 else 0
    recall = len(true_positives) / len(patch_files) if len(patch_files) > 0 else 0

    return precision, recall


def format_markdown_code_block(text, language=''):
    text = str(text).replace('```', '\\`\\`\\`')
    return f"```{language}\n{text}\n```"


def write_markdown(path, name, data):
    problem_statement = format_markdown_code_block(data['problem_statement'])
    hint = format_markdown_code_block(data['hints_text'])
    text = f"""# {data['instance_id']}

* repo: {data['repo']}
* base_commit: `{data['base_commit']}`

## Problem statement

{problem_statement}

## Hint

{hint}

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


def write_file(path, text):
    with open(path, 'w') as f:
        f.write(text)
        print(f"File '{path}' was saved", file=sys.stderr)


def benchmark_retrieve(pipeline_setup: IngestionPipelineSetup, benchmark_run: str, path: str, repo_name: str, row_data: dict, commit: str):
    perist_dir = f"/tmp/repos/{repo_name}-storage/{pipeline_setup.name}"
    if not os.path.exists(perist_dir):
        os.makedirs(perist_dir)

    print(f"Running benchmark for {pipeline_setup.name} on {repo_name} at commit {commit}")

    vector_store_name = f"{repo_name}-{commit}-{pipeline_setup.name}"
    db = get_chroma_db(vector_store_name, perist_dir)
    vectors, indexed_tokens = None, None

    try:
        chroma_collection = db.get_collection("files")
        vector_store = ChromaVectorStore(chroma_collection)
        retriever = LlamaIndexCodeSnippetRetriever.from_vector_store(vector_store, pipeline_setup.embed_model)
    except Exception as e:
        chroma_collection = None

    if not chroma_collection:
        chroma_collection = db.create_collection("files")
        vector_store = ChromaVectorStore(chroma_collection)
        retriever = LlamaIndexCodeSnippetRetriever.from_pipeline_setup(
                vector_store=vector_store,
                pipeline_setup=pipeline_setup,
                path=path,
                perist_dir=perist_dir,
            )

        vectors, indexed_tokens = retriever.run_index()

        try:
            upload_store(perist_dir, f"{repo_name}-{commit}-{pipeline_setup.name}")
        except Exception as e:
            print(f"Failed to upload store: {e}")

    patch_files = row_data["patch_files"]
    instance_id = row_data["instance_id"]

    query = row_data["problem_statement"]

    result = retriever.retrieve(query)

    no_of_patches = len(patch_files)

    found_patch_files = set()
    found_snippets = 0
    recommended_files = []

    top_file_pos = None
    file_pos = 0
    min_pos = None
    max_pos = None
    sum_pos = 0

    md = ""

    md += "\n\n## Expected patch\n\n"
    md += format_markdown_code_block(row_data["patch"], language='diff')

    expected_diffs = 0

    snippet_table_md = "\n\n## Retrieved code snippets\n\n"
    snippet_table_md += "| Position | File | Start line | End line | Tokens | Sum tokens |\n"
    snippet_table_md += "| -------- | ---- | ---------- | -------- | ------ | ---------- |\n"
    sum_tokens = 0

    any_found_context_length = None
    all_found_context_length = None

    snippet_md = ""

    for i, snippet in enumerate(result):
        md_line = ""

        tokenizer = get_tokenizer()
        tokens = len(tokenizer(snippet.content))

        if (sum_tokens + tokens >= 13000 and found_snippets == expected_diffs) or sum_tokens + tokens >= 50000:
            break

        sum_tokens += tokens

        snippet_hit = False
        snippet.path = snippet.path.replace(f"{path}/", "")

        if snippet.path not in recommended_files:
            recommended_files.append(snippet.path)
            file_pos += 1

        file_hit = snippet.path in row_data["patch_diff_details"]


        if file_hit:
            if not top_file_pos:
                top_file_pos = file_pos

            snippets_missing = False
            found_patch_files.add(snippet.path)
            diffs = row_data["patch_diff_details"][snippet.path]
            if snippet.start_line is not None and snippet.end_line is not None:
                for diff in diffs["diffs"]:

                    if snippet.start_line <= diff['start_line_old'] <= diff['end_line_old'] <= snippet.end_line:
                        found_snippets += 1
                        diff["pos"] = i+1
                        diff["context_length"] = sum_tokens

                        if min_pos is None:
                            min_pos = i+1
                        max_pos = i+1
                        sum_pos += i+1
                        snippet_hit = True

                        if any_found_context_length:
                            any_found_context_length = sum_tokens
                    elif "pos" not in diff:
                        snippets_missing = True

            if not snippets_missing and not all_found_context_length:
                all_found_context_length = sum_tokens

            if snippet_hit:
                md_line += f"| **-> {i+1} <-** "
            else:
                md_line += f"| {i+1} "

            md_line += f"| **{file_pos} {snippet.path}** "
        else:
            md_line += f"| {i+1} | {file_pos} {snippet.path} "

        if snippet.start_line is not None and snippet.end_line is not None:
            md_line += f"| {snippet.start_line} | {snippet.end_line}"
        else:
            md_line += "| - | - "

        md_line += f"| {tokens} | {sum_tokens} | \n"

        snippet_table_md += md_line

        if file_hit or i < 10:
            snippet_md += f"### {i + 1} - {snippet.path}:\n\n"

            if snippet.start_line and snippet.end_line:
                snippet_md += f"Start line: {snippet.start_line}, End line: {snippet.end_line}\n\n"

            snippet_md += f"```python\n{snippet.content}\n```\n"

    md += "\n\n## Expected file changes\n\n"
    md += "| File | Start line | End line | Found on position | Context length | \n"
    md += "| --- | --- | --- | ---- |\n"
    for file_path, diffs in row_data["patch_diff_details"].items():
        for diff in diffs["diffs"]:
            expected_diffs += 1
            md += f"| {file_path} | {diff['start_line_old']} | {diff['end_line_old']} | {diff.get('pos', '-')} | {diff.get('context_lenfth', '-')}\n"

    md += snippet_table_md

    missing_patch_files = len(patch_files) - len(found_patch_files)
    if missing_patch_files:
        md += "\n\n## Missing Patch Files\n"
        for i, file_path in enumerate(patch_files, start=1):
            md += f"\n * {i}: {file_path}"

    md += "\n\n## Problem Statement\n\n"
    md += format_markdown_code_block(row_data["problem_statement"])

    if row_data['hints_text']:
        md += "\n\n### Hint\n\n"
        md += format_markdown_code_block(row_data['hints_text'])

    md += f"""\n
## Patch

```diff
{row_data['patch']}
```

## Test Patch

```diff
{row_data['test_patch']}
```
"""

    md += "\n\n## Code snippets\n\n"
    md += snippet_md

    precision, recall = calculate_precision_recall(recommended_files, patch_files)

    avg_pos = sum_pos / no_of_patches if no_of_patches > 0 else None
    missing_snippets = expected_diffs - found_snippets

    with open(f"benchmark/reports/{benchmark_run}/summary.csv", "a") as f:
        print(f"{instance_id}: {pipeline_setup.name}:\n vectors: {vectors}\n tokens: {indexed_tokens}\n no_of_patches: {no_of_patches}\n any_found_context_length: {any_found_context_length} all_found_context_length: {any_found_context_length} \n \n avg_pos: {avg_pos}\n top_pos: {min_pos}\n worst_pos: {max_pos}\n top_file_pos: {top_file_pos}\n missing_snippets: {missing_snippets}\n missing_patch_files: {missing_patch_files}")
        csv.writer(f, delimiter=";").writerow([instance_id, pipeline_setup.name, vectors, indexed_tokens, no_of_patches, any_found_context_length, all_found_context_length, avg_pos, min_pos, max_pos, top_file_pos, missing_snippets, missing_patch_files])

    md_header = f"# {instance_id}"

    md_header += f"\n\n| **{row_data['repo']}** | `{row_data['base_commit']}` |\n"
    md_header += f"| ---- | ---- |\n"

    if vectors:
        md_header += f"| **Indexed vectors** | {vectors} |\n"

    if indexed_tokens:
        md_header += f"| **Indexed tokens** | {indexed_tokens} |\n"

    md_header += f"| **No of patches** | {no_of_patches} |\n"
    md_header += f"| **All found context length** | {all_found_context_length or '-'} |\n"
    md_header += f"| **Any found context length** | {all_found_context_length or '-'} |\n"
    md_header += f"| **Avg pos** | {avg_pos or '-'} |\n"
    md_header += f"| **Min pos** | {min_pos or '-'} |\n"
    md_header += f"| **Max pos** | {max_pos or '-'} |\n"
    md_header += f"| **Top file pos** | {top_file_pos or '-'} |\n"
    md_header += f"| **Missing snippets** | {missing_snippets} |\n"
    md_header += f"| **Missing patch files** | {missing_patch_files} |\n"

    md = md_header + md

    os.makedirs(f"benchmark/reports/{benchmark_run}/{instance_id}/patch_files", exist_ok=True)

    with open(f"benchmark/reports/{benchmark_run}/{instance_id}/report.md", "w") as f:
        f.write(md)

    for file_path in patch_files:
        shutil.copy(f"{path}/{file_path}", f"benchmark/reports/{benchmark_run}/{instance_id}/patch_files")

        file_name = file_path.split("/")[-1]

        split_report = report.generate_markdown(pipeline_setup.splitter, f"{path}/{file_path}")
        with open(f"benchmark/reports/{benchmark_run}/{instance_id}/patch_files/{file_name}_split.md", "w") as f:
            f.write(split_report)


def generate_summary(benchmark_run: str):
    with open(f"reports/{benchmark_run}/summary.csv", "r") as f:
        csv_file = csv.reader(f, delimiter=";")
        csv_file = sorted(csv_file, key=lambda x: x[0])

        sub_13 = 0
        sub_27 = 0
        sub_50 = 0

        md = ""
        top_file_position = 0

        for i, row in enumerate(csv_file):
            row.pop(1)
            if i > 0:
                row[0] = f"[{row[0]}](https://github.com/aorwall/moatless-tools/tree/main/benchmark/reports/{benchmark_run}/{row[0]}/report.md)"

            md += f"| {' | '.join(row)} |\n"
            if i == 0:
                md += f"| {' | '.join(['---' for _ in row])} |\n"
                continue

            if row[4] != "-":
                context_length = int(row[4])
                if context_length < 13000:
                    sub_13 += 1

                if context_length < 27000:
                    sub_27 += 1

                if context_length < 50000:
                    sub_50 += 1

        md += f"# Recall\n\n"
        md += "|     | 13k | 27k | 50k |\n"
        md += "| --- | --- | --- | --- |\n"

        sub_13_avg = round(sub_13 / (len(csv_file) -1) * 100, 2)
        sub_27_avg = round(sub_27 / (len(csv_file) -1) * 100, 2)
        sub_50_avg = round(sub_50 / (len(csv_file) -1) * 100, 2)
        md += f"| All | {sub_13_avg}% | {sub_27_avg}% | {sub_50_avg}% |\n\n"

        with open(f"reports/{benchmark_run}/summary.md", "w") as f:
            f.write(md)


def upload_store(store_path: str, file_name: str):
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        print("AZURE_STORAGE_CONNECTION_STRING is not set, cannot upload store.")
        return
    block_storage_client = BlobServiceClient.from_connection_string(connect_str)
    _blob_storage = block_storage_client.get_container_client(container="stores")

    zip_file_path = f"{store_path}/{file_name}"
    shutil.make_archive(zip_file_path, 'zip', f"{store_path}/chroma_db")

    blob_client = _blob_storage.get_blob_client(blob=file_name)
    with open(f"{zip_file_path}.zip", "rb") as data:
        blob_client.upload_blob(data)

    print(f"Uploaded {zip_file_path} to Azure Blob Storage with name {file_name}.")


def get_chroma_db(file_name: str, store_path: str):
    chromadb_path = f"{store_path}/{file_name}"
    if os.path.exists(chromadb_path):
        print(f"Using existing store at {chromadb_path}")
    elif download_store(file_name, store_path, chromadb_path):
        print(f"Use downloaded store at {chromadb_path}")

    return chromadb.PersistentClient(path=chromadb_path)


def download_store(file_name: str, store_path: str, chromadb_path: str):
    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        print("AZURE_STORAGE_CONNECTION_STRING is not set, cannot download store.")
        return False

    block_storage_client = BlobServiceClient.from_connection_string(connect_str)
    _blob_storage = block_storage_client.get_container_client(container="stores")

    zip_file_path = f"{store_path}/{file_name}.zip"

    blob_client = _blob_storage.get_blob_client(blob=zip_file_path)

    try:
        with open(zip_file_path, "wb") as data:
            blob_client.download_blob().readinto(data)
    except ResourceNotFoundError:
        return False
    except Exception as e:
        print(f"Failed to download {zip_file_path} from Azure Blob Storage: {e}")
        return False

    print(f"Downloaded {zip_file_path} from Azure Blob Storage.")

    shutil.unpack_archive(zip_file_path, chromadb_path)
    return True


if "__main__" == __name__:
    import dotenv
    dotenv.load_dotenv()
    generate_summary("princeton-nlp-SWE-bench_Lite-dev")