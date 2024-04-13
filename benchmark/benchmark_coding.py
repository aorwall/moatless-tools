import json
import os
import subprocess
from typing import List, Optional

import litellm
from dotenv import load_dotenv

from benchmark import swebench
from moatless.codeblocks.codeblocks import CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.utils import Colors
from moatless.coder import Coder
from moatless.types import ContextFile, Span
from moatless.utils.repo import setup_github_repo

import logging

load_dotenv("../.env")

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


logging.basicConfig(level=logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def benchmark_coding(
    instance_data: dict, benchmark_run: str = None, base_dir="/tmp/repos"
):
    repo_path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )
    print(f"{Colors.YELLOW}Cloned repo to path: {repo_path}{Colors.RESET}")

    parser = PythonParser()

    files = []

    for file_path in instance_data["patch_diff_details"].keys():
        with open(os.path.join(repo_path, file_path), "r") as f:
            file_content = f.read()

        codeblock = parser.parse(file_content)

        diff_spans = []
        for diff in instance_data["patch_diff_details"][file_path]["diffs"]:
            diff_spans.append(
                Span(
                    diff["start_line_old"],
                    diff.get("end_line_old", diff["start_line_old"]),
                )
            )

        context_spans = get_spans(codeblock, diff_spans)

        files.append(ContextFile(file_path=file_path, spans=context_spans))

    coder = Coder(
        repo_path=repo_path,
        requirement=instance_data["problem_statement"],
        session_id=benchmark_run,
        files=files,
        tags=[instance_data["instance_id"]],
    )

    coder.run()

    print(f"{Colors.BLUE}Expected patch:\n{instance_data['patch']}{Colors.RESET}")

    file_path = files[0].file_path

    diff = subprocess.run(
        ["git", "diff", file_path],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )

    if diff.stdout:
        print(f"{Colors.GREEN}Git diff:\n{diff.stdout}{Colors.RESET}")

        prediction = {
            "model_name_or_path": benchmark_run,
            "instance_id": instance_data["instance_id"],
            "model_patch": diff.stdout,
        }

        with open(f"{benchmark_run}_predictions.jsonl", "a") as file:
            json_string = json.dumps(prediction)
            file.write(json_string + "\n")

    else:
        print(f"{Colors.RED}No git diff{Colors.RESET}")


def get_spans(codeblock: CodeBlock, spans: List[Span]):

    # If block is withing one of the spans this should be returned
    within_span = is_block_within_spans(codeblock, spans)
    if within_span:
        return [Span(codeblock.start_line, codeblock.end_line)]

    # Return empty if there are no spans within the current block
    spans_within_block = get_spans_within_block(codeblock, spans)
    if not spans_within_block:
        return []

    # If there are ny indexed block the full block is returned
    # TODO: Check size and just pick X lines around the each span
    indexed_blocks = codeblock.get_indexed_blocks()
    if not indexed_blocks:
        return [Span(codeblock.start_line, codeblock.end_line)]

    spans_for_blocks = []
    for span_within_block in spans_within_block:
        indexed_blocks_within_span = find_indexed_blocks_by_span(
            codeblock, span_within_block
        )

        # If no indexed block within span, pick the span between the previous and next indexed block (or start/end of codeblock)
        if not indexed_blocks_within_span:
            start_line = codeblock.start_line
            end_line = codeblock.end_line
            for indexed_block in indexed_blocks:
                if indexed_block.start_line < span_within_block.start_line:
                    start_line = indexed_block.end_line + 1

                if indexed_block.end_line > span_within_block.end_line:
                    end_line = indexed_block.start_line - 1
                    break

            spans_for_blocks.append(Span(start_line, end_line))

        else:
            # Else, find the spans in the indexed blocks
            for indexed_block in indexed_blocks_within_span:
                spans_for_blocks.extend(get_spans(indexed_block, [span_within_block]))

    return spans_for_blocks


def get_spans_within_block(codeblock: CodeBlock, spans: List[Span]) -> List[Span]:
    return [
        span
        for span in spans
        if codeblock.start_line <= span.start_line
        and codeblock.end_line >= span.end_line
    ]


def find_indexed_blocks_by_span(codeblock: CodeBlock, span: Span):
    indexed_blocks = []
    for child in codeblock.children:
        if (
            span.start_line >= child.start_line
            and span.end_line <= child.end_line
            and child.is_indexed
        ):
            indexed_blocks.append(child)

        indexed_blocks.extend(find_indexed_blocks_by_span(child, span))
    return indexed_blocks


def is_block_within_spans(codeblock: CodeBlock, spans: List[Span]) -> Optional[Span]:
    for span in spans:
        if (
            span.start_line <= codeblock.start_line
            and span.end_line >= codeblock.end_line
        ):
            return span
    return None


def run_instances(benchmark_run: str):
    existing_patches = set()
    if os.path.exists(f"{benchmark_run}_predictions.jsonl"):
        with open(f"{benchmark_run}_predictions.jsonl", "r") as file:
            for line in file.readlines():
                prediction = json.loads(line)
                instance_id = prediction["instance_id"]
                existing_patches.add(instance_id)

    instances = swebench.get_instances(
        dataset_name="princeton-nlp/SWE-bench_Lite", split="test", data_dir="../data"
    )
    for i, instance_data in enumerate(instances):
        if instance_data["instance_id"] in existing_patches:
            print(
                f"{Colors.YELLOW}Skipping {instance_data['instance_id']}{Colors.RESET}"
            )
            continue

        print(
            f"{Colors.YELLOW}[{i}/{len(instances)}] Running {instance_data['instance_id']}{Colors.RESET}"
        )
        benchmark_coding(instance_data, benchmark_run=benchmark_run)


def run_instance(instance_id):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )
    benchmark_coding(instance_data, benchmark_run="testing")


if __name__ == "__main__":
    run_instances("moatless_gpt-4-turbo-2024-04-09_fully_assisted_context")

    # run_instance("astropy__astropy-6938")
