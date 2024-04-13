import json
import logging
import os

from dotenv import load_dotenv
from llama_index.core import get_tokenizer

from benchmark import swebench
from moatless.codeblocks import CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_by_block_paths
from moatless.codeblocks.utils import Colors
from moatless.constants import CHEAP_MODEL
from moatless.select_blocks import CodeBlockSelector, CodeSelectorResponse
from moatless.utils.repo import setup_github_repo


def _run_instance(instance_data: dict, base_dir: str = "/tmp/repos"):
    print(
        f"{Colors.BLUE}Running instance: {instance_data['instance_id']}{Colors.RESET}"
    )

    repo_path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )

    file_path = instance_data["patch_files"][0]

    result = {
        "instance_id": instance_data["instance_id"],
        "file_path": file_path,
    }

    parser = PythonParser()
    tokenize = get_tokenizer()
    with open(os.path.join(repo_path, file_path), "r") as f:
        content = f.read()

    codeblock = parser.parse(content)
    file_tokens = len(tokenize(content))
    result["file_tokens"] = file_tokens

    print(
        f"{Colors.BLUE}File path {file_path} with {file_tokens} (GPT) tokens{Colors.RESET}"
    )

    code_selector = CodeBlockSelector(model_name=CHEAP_MODEL, repo_path=repo_path)
    # TODO: Support multiple files with a for loop
    response = code_selector.select_blocks(
        instance_data["problem_statement"], file_path
    )

    print(f"{Colors.YELLOW}Thoughts: {response.thoughts or '??'}{Colors.RESET}")

    if not response.block_paths:
        print(f"{Colors.GREEN}No block selected{Colors.RESET}")
        result["status"] = "no_block_selected"
        result["tokens"] = result["file_tokens"]
        return result

    selected_content = print_by_block_paths(codeblock, response.block_paths)
    result["tokens"] = len(tokenize(selected_content))

    result["block_paths"] = response.block_paths

    missed_diffs = covered_all_blocks(instance_data, repo_path, response)
    result["missed_diffs"] = missed_diffs
    if not missed_diffs:
        print(f"{Colors.GREEN}All blocks covered{Colors.RESET}")
        result["status"] = "all_covered"
    else:
        print(f"{Colors.RED}Not all blocks covered{Colors.RESET}")
        result["status"] = "not_all_covered"

    return result


def run_instance(instance_id: str, base_dir: str = "/tmp/repos"):
    instance_data = swebench.get_instance(
        id=instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="dev",
        data_dir="../data",
    )
    _run_instance(instance_data, base_dir)


def run_instances(base_dir: str = "/tmp/repos", benchmark_id: str = None):
    if benchmark_id:
        print(f"{Colors.YELLOW}Running benchmark: {benchmark_id}{Colors.RESET}")
        if os.path.exists(f"benchmark_select_{benchmark_id}.json"):
            with open(f"benchmark_select_{benchmark_id}.json", "r") as f:
                benchmark_run_data = json.load(f)
        else:
            benchmark_run_data = {}

    success = 0
    instances = swebench.get_instances(
        dataset_name="princeton-nlp/SWE-bench_Lite", split="dev", data_dir="../data"
    )

    try:
        for instance_data in instances:
            if benchmark_id and instance_data["instance_id"] in benchmark_run_data:
                print(
                    f"{Colors.YELLOW}Skipping {instance_data['instance_id']}{Colors.RESET}"
                )
                result = benchmark_run_data[instance_data["instance_id"]]
                if result["status"] == "all_covered":
                    success += 1
                continue

            result = _run_instance(instance_data, base_dir)

            if result["status"] == "all_covered":
                success += 1

            if benchmark_id:
                print(
                    f"{Colors.YELLOW}Writing {instance_data['instance_id']}{Colors.RESET}"
                )
                benchmark_run_data[instance_data["instance_id"]] = result
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")

    if benchmark_id:
        with open(f"benchmark_select_{benchmark_id}.json", "w") as f:
            json.dump(benchmark_run_data, f, indent=2)

    print(f"Success: {success}, All: {len(instances)}")


def covered_diff(diff, selected_blocks):
    for selected_block in selected_blocks:
        if selected_block.start_line <= diff["start_line_old"] and (
            "end_line_old" not in diff
            or selected_block.end_line >= diff["end_line_old"]
        ):
            print(
                f"{Colors.GREEN}Diff covered {diff} by block path {selected_block.path_string()}{Colors.RESET}"
            )
            return True
    return False


def covered_all_blocks(instance_data, repo_path, response: CodeSelectorResponse):
    parser = PythonParser()

    missed_diffs = []
    for file_path, file_diff in instance_data["patch_diff_details"].items():

        with open(os.path.join(repo_path, file_path), "r") as f:
            file_content = f.read()

        codeblock = parser.parse(file_content)

        selected_blocks = []
        for block_path in response.block_paths:
            selected_block = codeblock.find_by_path(block_path)
            if selected_block:
                selected_blocks.append(selected_block)
                print(
                    f"{Colors.YELLOW}Block: {block_path} ({selected_block.start_line} - {selected_block.end_line}){Colors.RESET}"
                )
            else:
                print(f"{Colors.RED}Block not found: {block_path}{Colors.RESET}")

        for diff in file_diff["diffs"]:
            if not covered_diff(diff, selected_blocks):
                missed_diffs.append(diff)
                print(f"{Colors.RED}Diff not covered: {diff}{Colors.RESET}")

        if missed_diffs:
            print(
                f"{Colors.RED}Missing in patch {instance_data['patch']}{Colors.RESET}"
            )
            print(
                codeblock.to_tree(
                    only_identifiers=True,
                    include_line_numbers=True,
                    include_types=[CodeBlockType.CLASS, CodeBlockType.FUNCTION],
                )
            )

    return missed_diffs


if __name__ == "__main__":
    load_dotenv("../.env")
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    #    benchmark_reports()
    run_instances(benchmark_id="test")
    # run_instance("pvlib__pvlib-python-1606")
