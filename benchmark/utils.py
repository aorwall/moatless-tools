import json
import logging
import os
import re
import shutil
import sys
from typing import List, Optional

import faiss
import requests
from llama_index.core import get_tokenizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import NodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME

from moatless.code_index import CodeIndex
from moatless.codeblocks import CodeBlock
from moatless.codeblocks.codeblocks import Span
from moatless.ingestion import CodeBaseIngestionPipeline
from moatless.retriever import CodeSnippet, CodeSnippetRetriever
from moatless.store.simple_faiss import SimpleFaissVectorStore

logger = logging.getLogger(__name__)


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


def recall_report(row_data: dict, code_snippets: List[CodeSnippet], repo_path: str):
    report = {}
    report["instance_id"] = row_data["instance_id"]

    patch_files = row_data["patch_files"]
    no_of_patches = len(patch_files)

    found_patch_files = set()
    found_snippets = 0

    top_file_pos = None
    file_pos = 0
    min_pos = None
    sum_pos = 0

    sum_tokens = 0
    sum_file_tokens = 0

    snippet_reports = []
    file_reports = {}

    for i, snippet in enumerate(code_snippets):
        snippet_pos = i+1

        tokenizer = get_tokenizer()
        tokens = len(tokenizer(snippet.content))

        sum_tokens += tokens

        file_report = file_reports.get(snippet.file_path)
        if not file_report:
            file_pos += 1

            file_report = {
                "file_path": snippet.file_path,
                "position": file_pos
            }

            try:
                with open(f"{repo_path}/{snippet.file_path}", "r") as f:
                    file_tokens = len(tokenizer(f.read()))
                    sum_file_tokens += file_tokens
                    file_report["tokens"] = file_tokens
                    file_report["context_length"] = sum_file_tokens
            except Exception as e:
                logger.info(f"Failed to read file: {e}")
                file_report["error"] = str(e)
                file_report["snippet_id"] = snippet.id

            file_reports[snippet.file_path] = file_report

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

            found_patch_files.add(snippet.file_path)
            diffs = row_data["patch_diff_details"][snippet.file_path]
            for diff in diffs["diffs"]:
                if "file_pos" not in diff:
                    diff["file_pos"] = file_pos
                    diff["file_context_length"] = file_report["context_length"]

                if snippet.start_line and (snippet.start_line <= diff['start_line_old'] <= diff.get('end_line_old', diff['start_line_old']) <= snippet.end_line):
                    found_snippets += 1
                    diff["pos"] = snippet_pos
                    diff["context_length"] = sum_tokens

                    if "closest_snippet" in diff:
                        del diff["closest_snippet"]
                        del diff["closest_snippet_line_distance"]

                elif "pos" not in diff and snippet.start_line and snippet.end_line:
                    line_distance = min(abs(snippet.start_line - diff['start_line_old']), abs(snippet.end_line - diff.get('end_line_old', diff['start_line_old'])))

                    if "closest_snippet" not in diff or line_distance < diff["line_distance"]:
                        diff["closest_snippet_id"] = snippet.id
                        diff["closest_snippet_line_distance"] = line_distance

    report["patch_diff_details"] = row_data["patch_diff_details"]
    report["snippets"] = snippet_reports
    report["files"] = list(file_reports.values())

    return report


def diff_details(text: str):
    lines = text.split('\n')
    diffs = {}
    file_path = None
    for line in lines:
        if line.startswith('diff --git'):
            file_path = line.split(' ')[2][2:]  # Extract file name after 'b/'
            diffs[file_path] = {"diffs": []}
        elif line.startswith('@@'):
            # Extract the start line and size for old file from the chunk info
            match = re.search(r'\-(\d+),(\d+)', line)
            if match:
                start_line_old, size_old = match.groups()
                # Initialize tracking for the current diff chunk
                diffs[file_path]["diffs"].append({
                    "start_line_old": int(start_line_old),
                    "lines_until_first_change": 0
                })
        elif file_path and diffs[file_path]["diffs"]:
            current_diff = diffs[file_path]["diffs"][-1]

            if (line.startswith('+') or line.startswith('-')) and "lines_until_first_change" in current_diff:
                current_diff["start_line_old"] += current_diff["lines_until_first_change"]
                del current_diff["lines_until_first_change"]
            elif "lines_until_first_change" in current_diff:
                current_diff["lines_until_first_change"] += 1

            if line.startswith('-'):
                if "lines_until_last_minus" not in current_diff:
                    current_diff["lines_until_last_minus"] = 0
                else:
                    current_diff["lines_until_last_minus"] += 1

                current_diff["end_line_old"] = current_diff["start_line_old"] + current_diff["lines_until_last_minus"]
            elif not line.startswith('+') and "lines_until_last_minus" in current_diff:
                current_diff["lines_until_last_minus"] += 1

    # Final adjustments: remove temporary tracking keys
    for file_path, details in diffs.items():
        for diff in details["diffs"]:
            diff.pop("lines_until_last_minus", None)
            if "end_line_old" not in diff:
                diff["end_line_old"] = diff["start_line_old"]
    return diffs


def get_block_paths_from_diffs(codeblock: CodeBlock, diffs: dict) -> List[dict]:
    spans = []
    for diff in diffs:
        start_line = diff["start_line_old"]
        spans.append(Span(start_line, diff.get("end_line_old", start_line)))

    return [block.full_path() for block in codeblock.find_indexed_blocks_by_spans(spans)]


def get_blocks_from_diffs(codeblock: CodeBlock, diffs: dict) -> List[dict]:
    spans = []
    for diff in diffs:
        start_line = diff["start_line_old"]
        spans.append(Span(start_line, diff.get("end_line_old", start_line)))

    return [{
        "path": block.full_path() or "root",
        "block_id":  block.path_string(),
        "tokens":  block.sum_tokens(),
        "start_line":  block.start_line,
        "end_line":  block.end_line
    } for block in codeblock.find_indexed_blocks_by_spans(spans)]


def download_or_create_index(persist_dir: str,
                             ingestion_name: str,
                             repo_path: str,
                             repo_name: str,
                             base_commit: str,
                             embed_model: BaseEmbedding,
                             splitter: NodeParser) -> Optional[CodeIndex]:
    downloaded_existing_store = download_store(persist_dir, ingestion_name, repo_name, base_commit)

    # TODO: Remove this
    if not downloaded_existing_store:
        return None

    try:
        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)
    except:
        faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(1536))
        vector_store = SimpleFaissVectorStore(faiss_index)

    try:
        docstore = SimpleDocumentStore.from_persist_dir(persist_dir)
    except:
        docstore = SimpleDocumentStore()

    ingestion = CodeBaseIngestionPipeline(
        path=repo_path,
        vector_store=vector_store,
        docstore=docstore,
        embed_model=embed_model,
        num_workers=1,
    )

    if not downloaded_existing_store:
        ingestion.run()

        docstore.persist(persist_path=os.path.join(persist_dir, DEFAULT_PERSIST_FNAME))
        logger.info(f"Persisted docstore to {persist_dir}")
        vector_store.persist(persist_dir=persist_dir)
        logger.info(f"Persisted vector store to {persist_dir}")

    #try:
    #    upload_store(persist_dir, pipeline_name, f"{repo_name}-{commit}")
    #except Exception as e:
    #    logger.info(f"Failed to upload store: {e}")

    return ingestion.index()


def download_store(store_path: str, ingestion_name: str, repo_name: str, base_commit: str):
    repo_file_name = repo_name.replace("/", "__")

    base_url = "https://moatlesstools.blob.core.windows.net/stores"
    zip_file = f"{repo_file_name}-{base_commit}.zip"
    url = f"{base_url}/{ingestion_name}/{repo_name}-{base_commit}.zip"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_file, "wb") as data:
            for chunk in response.iter_content(chunk_size=8192):
                data.write(chunk)

    except requests.exceptions.HTTPError as e:
        logger.info(f"HTTP Error while fetching {zip_file}: {e}")
        return False
    except Exception as e:
        logger.info(f"Failed to download {zip_file}: {e}")
        return False

    shutil.unpack_archive(zip_file, store_path)

    logger.info(f"Downloaded {zip_file} from Azure Blob Storage.")

    os.remove(f"{zip_file}")

    return True

