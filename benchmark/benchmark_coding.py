import json
import os
import subprocess
from typing import List, Optional

import litellm
from dotenv import load_dotenv

from benchmark import swebench
from benchmark.swebench import get_spans
from moatless.codeblocks.codeblocks import CodeBlock
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.utils import Colors
from moatless.coder import Coder
from moatless.settings import Settings
from moatless.types import ContextFile, Span
from moatless.utils.repo import setup_github_repo

import logging

load_dotenv("../.env")

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]


Settings.coder.enable_chain_of_thought = False


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
                    start_line=diff["start_line_old"],
                    end_line=diff.get("end_line_old", diff["start_line_old"]),
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
    else:
        print(f"{Colors.RED}No git diff found{Colors.RESET}")

    prediction = {
        "model_name_or_path": benchmark_run,
        "instance_id": instance_data["instance_id"],
        "model_patch": diff.stdout,
    }

    with open(f"{benchmark_run}_predictions.jsonl", "a") as file:
        json_string = json.dumps(prediction)
        file.write(json_string + "\n")


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
        try:
            benchmark_coding(instance_data, benchmark_run=benchmark_run)
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
            with open(f"{benchmark_run}_errors.txt", "a") as file:
                file.write(f"{instance_data['instance_id']}: {e}\n")


def run_instance(instance_id):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )
    benchmark_coding(instance_data, benchmark_run="testing")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    run_instances("moatless_gpt-4-turbo-2024-04-15_fully_assisted_context")

    # run_instance("django__django-14752")
