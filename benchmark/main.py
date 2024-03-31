import json
import logging
import os
import shutil

import faiss
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from llama_index.core import get_tokenizer
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME
from llama_index.embeddings.openai import OpenAIEmbedding

from moatless.retrievers.golden_retriever import IngestionPipelineSetup, GoldenRetriever
from moatless.retrievers.ingestion import Ingestion
from moatless.splitters import report
from moatless.splitters.epic_split import EpicSplitter, CommentStrategy
from moatless.store.simple_faiss import SimpleFaissVectorStore

logger = logging.getLogger(__name__)


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
        logger.info(f"File '{path}' was saved")


def initiate_index(path: str, index_name: str, persist_dir: str = "/tmp/.storage"):
    pipeline_setup = IngestionPipelineSetup(
        name="epic-splitter-v4--100-750--comment-associate--text-embedding-3-small--1536",
        splitter=EpicSplitter(min_chunk_size=100, chunk_size=750, language="python", comment_strategy=CommentStrategy.ASSOCIATE),
        embed_model=OpenAIEmbedding(model="text-embedding-3-small")
    )

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    downloaded_existing_store = download_store(persist_dir, pipeline_setup.name, index_name)

    try:
        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
    except:
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
        vector_store = SimpleFaissVectorStore(faiss_index)

    try:
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
    except:
        docstore = SimpleDocumentStore()

    if not downloaded_existing_store:
        ingestion = Ingestion(
            vector_store=vector_store,
            docstore=docstore,
            pipeline_setup=pipeline_setup,
            path=path,
            perist_dir=persist_dir
        )
        vectors, indexed_tokens = ingestion.run()
        print(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")

        docstore.persist(persist_path=os.path.join(persist_dir, DEFAULT_PERSIST_FNAME))
        logger.info(f"Persisted docstore to {persist_dir}")
        vector_store.persist(persist_dir=persist_dir)
        logger.info(f"Persisted vector store to {persist_dir}")

        try:
            upload_store(persist_dir, pipeline_setup.name, index_name)
        except Exception as e:
            logger.info(f"Failed to upload store: {e}")

    retriever = GoldenRetriever(
        vector_store=vector_store,
        docstore=docstore,
        embed_model=pipeline_setup.embed_model)

    return retriever


def benchmark_retrieve(pipeline_setup: IngestionPipelineSetup, benchmark_run: str, path: str, repo_name: str, row_data: dict, commit: str, report_dir='benchmark/reports'):
    persist_dir = f"/tmp/repos/{repo_name}-storage/{pipeline_setup.name}"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    logger.info(f"Running benchmark for {pipeline_setup.name} on {repo_name} at commit {commit}")

    downloaded_existing_store = download_store(persist_dir, pipeline_setup.name, f"{repo_name}-{commit}")

    try:
        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
    except:
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
        vector_store = SimpleFaissVectorStore(faiss_index)

    vectors, indexed_tokens = None, None
    try:
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
    except:
        docstore = SimpleDocumentStore()

    if not downloaded_existing_store:
        ingestion = Ingestion(
            vector_store=vector_store,
            docstore=docstore,
            pipeline_setup=pipeline_setup,
            path=path,
            perist_dir=persist_dir
        )
        vectors, indexed_tokens = ingestion.run()

        row_data["vectors"] = vectors
        row_data["indexed_tokens"] = indexed_tokens

        docstore.persist(persist_path=os.path.join(persist_dir, DEFAULT_PERSIST_FNAME))
        logger.info(f"Persisted docstore to {persist_dir}")
        vector_store.persist(persist_dir=persist_dir)
        logger.info(f"Persisted vector store to {persist_dir}")

        try:
            upload_store(persist_dir, pipeline_setup.name, f"{repo_name}-{commit}")
        except Exception as e:
            logger.info(f"Failed to upload store: {e}")
    else:
        row_data["vectors"] = len(docstore.docs)

    retriever = GoldenRetriever(
        vector_store=vector_store,
        docstore=docstore,
        embed_model=pipeline_setup.embed_model)

    patch_files = row_data["patch_files"]
    instance_id = row_data["instance_id"]

    query = row_data["problem_statement"]

    result = retriever.retrieve(query)

    no_of_patches = len(patch_files)

    found_patch_files = set()
    found_snippets = 0

    top_file_pos = None
    file_pos = 0
    min_pos = None
    max_pos = None
    sum_pos = 0

    md = ""

    md += "\n\n## Expected patch\n\n"
    md += format_markdown_code_block(row_data["patch"], language='diff')

    missing_snippets = 0

    snippet_table_md = "\n\n## Retrieved code snippets\n\n"
    snippet_table_md += "| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |\n"
    snippet_table_md += "| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |\n"
    sum_tokens = 0
    sum_file_tokens = 0

    any_found_context_length = None
    all_found_context_length = None

    snippet_md = ""

    snippet_reports = []
    file_reports = {}

    for i, snippet in enumerate(result):
        snippet_pos = i+1
        md_line = ""

        tokenizer = get_tokenizer()
        tokens = len(tokenizer(snippet.content))

        if sum_file_tokens + tokens >= 200000:
            break

        sum_tokens += tokens

        file_report = file_reports.get(snippet.file_path)
        if not file_report:
            file_pos += 1

            file_report = {
                "file_path": snippet.file_path.replace(f"{path}/", ""),
                "position": file_pos
            }

            try:
                with open(snippet.file_path, "r") as f:
                    file_tokens = len(tokenizer(f.read()))
                    sum_file_tokens += file_tokens
                    file_report["tokens"] = file_tokens
                    file_report["context_length"] = sum_file_tokens
            except Exception as e:
                logger.info(f"Failed to read file: {e}")
                file_report["error"] = str(e)
                file_report["snippet_id"] = snippet.id

            file_reports[snippet.file_path] = file_report

        snippet_hit = False
        snippet.file_path = file_report["file_path"]

        file_hit = snippet.file_path in row_data["patch_diff_details"]

        snippet_report = {
            "position": snippet_pos,
            "id": snippet.id,
            "distance": snippet.distance,
            "file_path": snippet.file_path,
            "start_line": snippet.start_line,
            "end_line": snippet.end_line,
            "tokens": tokens,
            "context_length": sum_tokens,
        }
        snippet_reports.append(snippet_report)

        if file_hit:
            if not top_file_pos:
                top_file_pos = file_pos

            snippet_report["file_pos"] = file_pos

            snippets_missing = False
            found_patch_files.add(snippet.file_path)
            diffs = row_data["patch_diff_details"][snippet.file_path]
            if snippet.start_line is not None and snippet.end_line is not None:
                for diff in diffs["diffs"]:
                    if "file_pos" not in diff:
                        diff["file_pos"] = file_pos
                        diff["file_context_length"] = file_report["context_length"]

                    if (snippet.start_line <= diff['start_line_old'] <= diff.get('end_line_old', diff['start_line_old']) <= snippet.end_line):
                        found_snippets += 1
                        diff["pos"] = snippet_pos
                        diff["context_length"] = sum_tokens

                        if "closest_snippet" in diff:
                            del diff["closest_snippet"]
                            del diff["closest_snippet_line_distance"]

                        if min_pos is None:
                            min_pos = snippet_pos
                        max_pos = snippet_pos
                        sum_pos += snippet_pos
                        snippet_hit = True

                        if not any_found_context_length:
                            any_found_context_length = sum_tokens
                    elif "pos" not in diff:
                        line_distance = min(abs(snippet.start_line - diff['start_line_old']), abs(snippet.end_line - diff.get('end_line_old', diff['start_line_old'])))

                        if "closest_snippet" not in diff or line_distance < diff["line_distance"]:
                            diff["closest_snippet_id"] = snippet.id
                            diff["closest_snippet_line_distance"] = line_distance

                        snippets_missing = True

            if not snippets_missing and not all_found_context_length:
                all_found_context_length = sum_tokens

            if snippet_hit:
                md_line += f"| **-> {i+1} <-** "
            else:
                md_line += f"| {i+1} "

            md_line += f"| **{file_pos} {snippet.file_path}** "
        else:
            md_line += f"| {i+1} | {file_pos} {snippet.file_path} "

        if snippet.start_line is not None and snippet.end_line is not None:
            md_line += f"| {snippet.start_line} | {snippet.end_line}"
        else:
            md_line += "| - | - "

        md_line += f"| {tokens} | {sum_tokens} | {sum_file_tokens} | \n"

        if (sum_tokens + tokens < 13000 and not all_found_context_length) or sum_file_tokens + tokens <= 200000:
            snippet_table_md += md_line

        if file_hit or i < 10:
            snippet_md += f"### {i + 1} - {snippet.file_path}:\n\n"

            if snippet.start_line and snippet.end_line:
                snippet_md += f"Start line: {snippet.start_line}, End line: {snippet.end_line}\n\n"

            snippet_md += f"```python\n{snippet.content}\n```\n"

    row_data["snippets"] = snippet_reports
    row_data["files"] = list(file_reports.values())

    try:
        json.dump(row_data, open(f"{report_dir}/{benchmark_run}/{instance_id}/data.json", "w"), indent=2)
    except Exception as e:
        logger.info(f"Failed to write data.json: {e}")

    md += "\n\n## Expected file changes\n\n"
    md += "| File | Start line | End line | Found on position | Found file position | Context length |\n"
    md += "| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |\n"
    for file_path, diffs in row_data["patch_diff_details"].items():
        for diff in diffs["diffs"]:
            if "post" not in diff:
                missing_snippets += 1
            md += f"| {file_path} | {diff['start_line_old']} | {diff.get('end_line_old', '-')} | {diff.get('pos', '-')} | {diff.get('file_pos', '-')} | {diff.get('context_length', '-')}\n"

    md += "\n\n## Problem Statement\n\n"
    md += format_markdown_code_block(row_data["problem_statement"])

    md += snippet_table_md

    missing_patch_files = len(patch_files) - len(found_patch_files)
    if missing_patch_files:
        md += "\n\n## Missing Patch Files\n"
        for i, file_path in enumerate(patch_files, start=1):
            md += f"\n * {i}: {file_path}"

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

    avg_pos = sum_pos / no_of_patches if no_of_patches > 0 else None

    md_header = f"# {instance_id}"

    md_header += f"\n\n| **{row_data['repo']}** | `{row_data['base_commit']}` |\n"
    md_header += f"| ---- | ---- |\n"

    if vectors:
        md_header += f"| **Indexed vectors** | {vectors} |\n"

    if indexed_tokens:
        md_header += f"| **Indexed tokens** | {indexed_tokens} |\n"

    md_header += f"| **No of patches** | {no_of_patches} |\n"
    md_header += f"| **All found context length** | {all_found_context_length or '-'} |\n"
    md_header += f"| **Any found context length** | {any_found_context_length or '-'} |\n"
    md_header += f"| **Avg pos** | {avg_pos or '-'} |\n"
    md_header += f"| **Min pos** | {min_pos or '-'} |\n"
    md_header += f"| **Max pos** | {max_pos or '-'} |\n"
    md_header += f"| **Top file pos** | {top_file_pos or '-'} |\n"
    md_header += f"| **Missing snippets** | {missing_snippets} |\n"
    md_header += f"| **Missing patch files** | {missing_patch_files} |\n"

    md = md_header + md

    os.makedirs(f"{report_dir}/{benchmark_run}/{instance_id}/patch_files", exist_ok=True)

    with open(f"{report_dir}/{benchmark_run}/{instance_id}/report.md", "w") as f:
        f.write(md)

    for file_path in patch_files:
        if not os.path.exists(f"{path}/{file_path}"):
            continue
        shutil.copy(f"{path}/{file_path}", f"{report_dir}/{benchmark_run}/{instance_id}/patch_files")

        file_name = file_path.split("/")[-1]

        split_report = report.generate_markdown(pipeline_setup.splitter, f"{path}/{file_path}")
        with open(f"{report_dir}/{benchmark_run}/{instance_id}/patch_files/{file_name}_split.md", "w") as f:
            f.write(split_report)


def upload_store(store_path: str, ingestion_name: str, file_name: str):
    shutil.make_archive(file_name, 'zip', store_path)

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

    os.remove(f"{file_name}.zip")

    logger.info(f"Uploaded {blob_name} to Azure Blob Storage.")


def download_store(store_path: str, ingestion_name: str, file_name: str):
    zip_file = f"zips/{file_name}.zip"
    if os.path.exists(zip_file):
        logger.info(f"Found {zip_file} locally, skipping download.")
        shutil.unpack_archive(zip_file, store_path)
        return True

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connect_str:
        logger.info("AZURE_STORAGE_CONNECTION_STRING is not set, cannot download store.")
        return False

    block_storage_client = BlobServiceClient.from_connection_string(connect_str)
    _blob_storage = block_storage_client.get_container_client(container="stores")

    blob_name = f"{ingestion_name}/{zip_file}"

    blob_client = _blob_storage.get_blob_client(blob=blob_name)

    try:
        with open(zip_file, "wb") as data:
            blob_client.download_blob().readinto(data)
    except ResourceNotFoundError:
        logger.info(f"{blob_name} not found in Azure Blob Storage")
        return False
    except Exception as e:
        logger.info(f"Failed to download {zip_file} from Azure Blob Storage: {e}")
        return False

    shutil.unpack_archive(zip_file, store_path)

    logger.info(f"Downloaded {zip_file} from Azure Blob Storage.")

    os.remove(f"{zip_file}")

    return True


if "__main__" == __name__:
    import dotenv
    dotenv.load_dotenv()
