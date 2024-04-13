import csv
import json
import os
from typing import List

import yaml
from litellm import completion
from llama_index.core import get_tokenizer

from benchmark.swebench import get_instances
from benchmark.utils import diff_details
from moatless.code_graph import CodeGraph
from moatless.codeblocks import CodeBlock
from moatless.codeblocks.codeblocks import CodeBlockType
from moatless.types import Span
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.print_block import print_by_block_paths
from moatless.utils.repo import setup_github_repo

from moatless.verify.python_test_parser import extract_test_details, find_failed_tests


def read_pipelines(dir: str):
    pipelines = {}

    for subdir in os.listdir(dir):
        file = os.path.join(dir, subdir, "state.json")
        if os.path.exists(file):
            with open(file, "r") as f:
                pipeline = json.load(f)
            pipelines[subdir] = pipeline

    return pipelines


def read_scorecards(dir: str):
    with open(os.path.join(dir, "scorecards.json"), "r") as f:
        scoreboard = json.load(f)
    return scoreboard


def read_predictions(dir: str):
    file_path = os.path.join(dir, "predictions.jsonl")
    predictions = {}
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            predictions[data["instance_id"]] = data
            predictions[data["instance_id"]]["diff_details"] = diff_details(
                data["model_patch"]
            )
    return predictions


def generate_report(dir: str, repo_dir: str = "/tmp/repos", overwrite: bool = False):
    pipelines = read_pipelines(dir)

    scoreboard = read_scorecards(dir)

    instances = get_instances(
        split="test", dataset_name="princeton-nlp/SWE-bench_Lite", data_dir="../data"
    )
    instance_by_id = {instance["instance_id"]: instance for instance in instances}

    predictions = read_predictions(dir)

    tokenizer = get_tokenizer()

    results = []

    for instance in scoreboard:
        instance_id = instance["instance_id"]
        result_file = f"{dir}/{instance_id}/result.json"
        if os.path.exists(result_file) and not overwrite:
            print(f"Skipping {instance_id}")

            with open(result_file, "r") as f:
                result = json.load(f)
            results.append(result)
            continue

        instance_data = instance_by_id[instance_id]
        pipeline = pipelines[instance_id]

        if "test_results" in instance:
            successful_tests = len(
                instance["test_results"]["success"]["FAIL_TO_PASS"]
            ) + len(instance["test_results"]["success"]["PASS_TO_PASS"])
            failed_tests = len(
                instance["test_results"]["failure"]["FAIL_TO_PASS"]
            ) + len(instance["test_results"]["failure"]["PASS_TO_PASS"])
        else:
            successful_tests = 0
            failed_tests = 0

        repo_path = setup_github_repo(
            repo=instance_data["repo"],
            base_commit=instance_data["base_commit"],
            base_dir=repo_dir,
        )

        patch_file = instance_data["patch_files"][
            0
        ]  # TODO: Only suppporting one file ATM

        golden_patch = instance_data["patch_diff_details"][patch_file]

        if patch_file in predictions[instance_id]["diff_details"]:
            prediction = predictions[instance_id]["diff_details"][patch_file]
        else:
            prediction = {"diffs": []}

        code_graph = CodeGraph()

        def add_to_graph(codeblock: CodeBlock):
            code_graph.add_to_graph(patch_file, codeblock)

        parser = PythonParser(index_callback=add_to_graph)

        with open(f"{repo_path}/{patch_file}", "r") as f:
            content = f.read()

        file_tokens = len(tokenizer(content))

        codeblock = parser.parse(content)

        expected_blocks = get_blocks_from_patch(codeblock, golden_patch)
        expected_block_blockpaths = [block["path"] for block in expected_blocks]
        print_content = print_by_block_paths(codeblock, expected_block_blockpaths)
        content_file = f"{dir}/{instance['instance_id']}/expected_blocks.py"
        with open(content_file, "w") as f:
            f.write(print_content)

        actual_blocks = get_blocks_from_patch(codeblock, prediction)

        selected_blocks = []
        steps = pipeline["steps"]
        for step in steps:
            if step["action"] == "select_blocks":
                selected_blocks = step["response"]["block_paths"]

        print_content = print_by_block_paths(codeblock, selected_blocks)
        content_file = f"{dir}/{instance['instance_id']}/selected_blocks.py"
        with open(content_file, "w") as f:
            f.write(print_content)
        selected_tokens = len(tokenizer(print_content))

        selected_blocks_and_related_blocks = []
        for selected_block_path in selected_blocks:
            selected_blocks_and_related_blocks.append(selected_block_path)
            related_blocks = code_graph.find_relationships(
                patch_file, selected_block_path
            )
            if related_blocks:
                for related_block in related_blocks:
                    if related_block.type in [
                        CodeBlockType.CLASS,
                        CodeBlockType.MODULE,
                    ]:
                        for child in related_block.children:
                            if (
                                not child.is_indexed
                                and child.full_path()
                                not in selected_blocks_and_related_blocks
                            ):
                                selected_blocks_and_related_blocks.append(
                                    child.full_path()
                                )

                    elif (
                        related_block.full_path()
                        not in selected_blocks_and_related_blocks
                    ):
                        selected_blocks_and_related_blocks.append(
                            related_block.full_path()
                        )

        print_content = print_by_block_paths(
            codeblock, selected_blocks_and_related_blocks
        )
        content_file = (
            f"{dir}/{instance['instance_id']}/selected_blocks_with_relations.py"
        )
        with open(content_file, "w") as f:
            f.write(print_content)

        relations_tokens = len(tokenizer(print_content))

        not_selected_blocks = []
        not_selected_with_relationships = []
        missing_blocks = []
        for expected_block in expected_blocks:
            found_blocks = [
                block
                for block in actual_blocks
                if block["block_id"] == expected_block["block_id"]
            ]
            if not found_blocks:
                missing_blocks.append(expected_block)

            found_selected_blocks = [
                block_path
                for block_path in selected_blocks
                if block_path == expected_block["path"]
            ]
            if not found_selected_blocks:
                not_selected_blocks.append(expected_block)

            found_selected_with_relationship = [
                block_path
                for block_path in selected_blocks_and_related_blocks
                if block_path == expected_block["path"]
            ]
            if not found_selected_with_relationship:
                not_selected_with_relationships.append(expected_block)

        missing_patches = get_missing_patches(prediction, golden_patch)

        min_line_number = 1000000
        max_line_number = 0
        if len(missing_patches) > 0:
            for diff in missing_patches:
                if diff["start_line_old"] < min_line_number:
                    min_line_number = diff["start_line_old"]
                if diff["end_line_old"] > max_line_number:
                    max_line_number = diff["end_line_old"]

        min_line_in_prediction = 1000000
        max_line_in_prediction = 0
        for diff in prediction["diffs"]:
            if diff["start_line_old"] < min_line_in_prediction:
                min_line_in_prediction = diff["start_line_old"]
            if diff["end_line_old"] > max_line_in_prediction:
                max_line_in_prediction = diff["end_line_old"]

        status = ""
        if "RESOLVED_FULL" in instance["statuses"]:
            status = "Resolved"
        elif "applied" in instance["statuses"]:
            status = "Applied"
        elif "install_fail" in instance["statuses"]:
            status = "Install Fail"
        elif "generated" in instance["statuses"]:
            status = "Generated"
        elif "not_generated" in instance["statuses"]:
            status = "Not Generated"

        benchmark_name = predictions[instance_id]["model_name_or_path"]
        test_log_file = os.path.join(
            dir, "logs", f"{instance['instance_id']}.{benchmark_name}.eval.log"
        )
        if os.path.exists(test_log_file):
            with open(test_log_file, "r") as f:
                test_logs = f.read()
        else:
            test_logs = None

        result = {
            "instance_id": instance_id,
            "repo": instance_data["repo"],
            "base_commit": instance_data["base_commit"],
            "patch_files": instance.get("patch_files"),
            "file_tokens": file_tokens,
            "problem_statement": instance_data["problem_statement"],
            "model_patch": predictions[instance_id]["model_patch"],
            "expected_patch": instance_data["patch"],
            "status": status,
            "steps": pipeline["steps"],
            "missing_patches": len(missing_patches),
            "updated_blocks": actual_blocks,
            "expected_blocks": expected_blocks,
            "missing_blocks": missing_blocks,
            "selected_blocks": selected_blocks,
            "selected_tokens": selected_tokens,
            "not_selected_blocks": not_selected_blocks,
            "selected_blocks_and_related_blocks": selected_blocks_and_related_blocks,
            "not_selected_with_relationships": not_selected_with_relationships,
            "relations_tokens": relations_tokens,
            "min_line_number": min_line_number if min_line_number < 1000000 else "",
            "max_line_number": max_line_number if max_line_number > 0 else "",
            "min_line_in_prediction": (
                min_line_in_prediction if min_line_in_prediction < 1000000 else ""
            ),
            "max_line_in_prediction": (
                max_line_in_prediction if max_line_in_prediction > 0 else ""
            ),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "test_logs": test_logs,
        }

        results.append(result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=4)

        markdown = generate_markdown(result)
        report_file = f"{dir}/{instance['instance_id']}/report.md"
        with open(report_file, "w") as f:
            f.write(markdown)

    csv_file = f"{dir}/results.csv"
    with open(csv_file, "w") as f:
        f.write(
            "Instance ID;Status;File tokens;Missing patches;Missing blocks;Missing block IDs;Selected tokens;Not selected blocks;Not selected block IDs;Relations tokens;Not selected with relationships;Min line number;Max line number;Min line in prediction;Max line in prediction;Successful tests;Failed tests\n"
        )

        for result in results:
            block_ids = [
                block["block_id"]
                for block in result["missing_blocks"]
                if block["block_id"]
            ]
            not_selected_blocks = [
                block["block_id"]
                for block in result["not_selected_blocks"]
                if block["block_id"]
            ]
            not_selected_with_relationships = [
                block["block_id"]
                for block in result["not_selected_with_relationships"]
                if block["block_id"]
            ]

            try:
                f.write(
                    f"{result['instance_id']};{result['status']};{result['file_tokens']};{result['missing_patches']}; {len(block_ids)}; {','.join(block_ids)};{result['selected_tokens']};{len(not_selected_blocks)};{','.join(not_selected_blocks)};{result['relations_tokens']};{len(not_selected_with_relationships)};{','.join(not_selected_with_relationships)};{result['min_line_number']};{result['max_line_number']};{result['min_line_in_prediction']};{result['max_line_in_prediction']};{result['successful_tests']};{result['failed_tests']}\n"
                )
            except Exception as e:
                print(e)
                raise e

    csv_file = f"{dir}/selection_result.csv"
    with open(csv_file, "w") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Instance ID",
                "Status",
                "File tokens",
                "Updated blocks",
                "Updated block IDs",
                "Expected blocks",
                "Expected block IDs",
                "Missing blocks",
                "Missing block IDs",
                "Selected blocks",
                "Selected block IDs",
                "Selected tokens",
                "Not selected blocks",
                "Not selected block IDs",
                "Selected blocks with relationships",
                "Selected block IDs with relationships",
                "Selected with relationship tokens",
                "Not selected with relationships",
                "Not selected block IDs with relationships",
            ]
        )
        for result in results:
            updated_blocks = [
                block["block_id"]
                for block in result["updated_blocks"]
                if block["block_id"]
            ]
            expected_blocks = [
                block["block_id"]
                for block in result["expected_blocks"]
                if block["block_id"]
            ]

            missing_blocks = [
                block["block_id"]
                for block in result["missing_blocks"]
                if block["block_id"]
            ]

            selected_blocks = [
                ".".join(block_path) for block_path in result["selected_blocks"]
            ]
            not_selected_blocks = [
                block["block_id"]
                for block in result["not_selected_blocks"]
                if block["block_id"]
            ]

            selected_blocks_and_related_blocks = [
                ".".join(block_path)
                for block_path in result["selected_blocks_and_related_blocks"]
            ]
            not_selected_with_relationships = [
                block["block_id"]
                for block in result["not_selected_with_relationships"]
                if block["block_id"]
            ]

            updated_blocks.sort()
            expected_blocks.sort()
            missing_blocks.sort()
            selected_blocks.sort()
            not_selected_blocks.sort()
            selected_blocks_and_related_blocks.sort()
            not_selected_with_relationships.sort()

            writer.writerow(
                [
                    result["instance_id"],
                    result["status"],
                    result["file_tokens"],
                    len(updated_blocks),
                    "\n".join(updated_blocks),
                    len(expected_blocks),
                    "\n".join(expected_blocks),
                    len(missing_blocks),
                    "\n".join(missing_blocks),
                    len(selected_blocks),
                    "\n".join(selected_blocks),
                    result["selected_tokens"],
                    len(not_selected_blocks),
                    "\n".join(not_selected_blocks),
                    len(selected_blocks_and_related_blocks),
                    "\n".join(selected_blocks_and_related_blocks),
                    result["relations_tokens"],
                    len(not_selected_with_relationships),
                    "\n".join(not_selected_with_relationships),
                ]
            )
    _generate_markdown_mkdocs(dir, results)


def _generate_markdown_mkdocs(dir: str, results: List[dict]):
    status_groups = {}
    for result in results:
        status = result["status"]
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(result)

    statuses = ["Resolved", "Applied", "Install Fail", "Generated", "Not Generated"]

    mkdocs_config = {
        "site_name": "Benchmark",
        "docs_dir": ".",
        "nav": [{"Home": "index.md"}],
        "theme": {
            "name": "material",
        },
    }

    markdown = "# Summary\n"
    for status in statuses:
        results = status_groups.get(status, [])

        if not results:
            continue

        section = {status: []}

        markdown += f"\n## {status}\n"
        markdown += (
            "\n| Instance ID | Missing patches | Successful tests | Failed tests |\n"
        )
        markdown += "| --- | --- | --- | --- |\n"
        for result in results:
            doc_path = f"{result['instance_id']}/report.md"
            section[status].append({result["instance_id"]: doc_path})
            markdown += f"| [{result['instance_id']}]({doc_path}) | {result['missing_patches']} | {result['successful_tests']} | {result['failed_tests']} |\n"

        mkdocs_config["nav"].append(section)

    with open(f"{dir}/index.md", "w") as f:
        f.write(markdown)

    mkdocs_yaml = yaml.dump(mkdocs_config, sort_keys=False)
    with open(f"{dir}/mkdocs.yml", "w") as f:
        f.write(mkdocs_yaml)

    return markdown


def generate_markdown(result: dict):
    markdown = f"# {result['instance_id']}\n"

    markdown += f"\n * Status: {result['status']}"
    markdown += f"\n * Missing patches: {result['missing_patches']}"
    markdown += f"\n * Successful tests: {result['successful_tests']}"
    markdown += f"\n * Failed tests: {result['failed_tests']}"

    markdown += f"\n\n## Problem statement"
    problem_statement = result["problem_statement"]
    problem_statement = problem_statement.replace("```", "``")
    markdown += f"\n```\n{problem_statement}\n```\n"

    markdown += f"\n\n## Patch"
    markdown += f"\n```diff\n{result['model_patch']}\n```\n"

    markdown += f"\n\n## Golden patch"
    markdown += f"\n```diff\n{result['expected_patch']}\n```\n"

    markdown += f"\n\n## Blocks"

    markdown += f"\n\n### Expected blocks"
    for block in result["expected_blocks"]:
        markdown += f"\n\n * `{block['block_id']}` ({block['start_line']} - {block['end_line']}, {block['tokens']} tokens)"

    markdown += f"\n\n### Updated blocks"
    for block in result["updated_blocks"]:
        markdown += f"\n\n * `{block['block_id']}` ({block['start_line']} - {block['end_line']}, {block['tokens']} tokens)"

    markdown += f"\n\n### Selected blocks"
    for block in result["selected_blocks"]:
        markdown += f"\n\n * `{'.'.join(block)}`"

    markdown += f"\n\n### Selected blocks with references"
    for block in result["selected_blocks_and_related_blocks"]:
        markdown += f"\n\n *  `{'.'.join(block)}`"

    markdown += f"\n\n## Steps"
    for step in result["steps"]:
        markdown += f"\n\n### {step['action']}"

        response = step.get("response")

        markdown += f"\n\n#### Token usage"
        prompt_tokens, completion_tokens, total_tokens = calculate_tokens(
            response["usage_stats"]
        )
        markdown += f"\n * LLM calls: {len(response['usage_stats'])}"
        markdown += f"\n * Prompt tokens: {prompt_tokens}"
        markdown += f"\n * Completion tokens: {completion_tokens}"
        markdown += f"\n * Total tokens: {total_tokens}"

        if "thoughts" in step:
            markdown += f"\n\n#### Thoughts"
            markdown += f"\n{step['thoughts']}"

        if step["action"] == "select_blocks":
            markdown += f"\n\n#### Selected blocks"
            for block_path in response["block_paths"]:
                block_path_str = ".".join(block_path)
                markdown += f"\n * `{block_path_str}`"

        if step["action"] == "plan":
            markdown += f"\n\n#### Tasks"

            i = 0
            for task in response["tasks"]:
                i += 1
                markdown += f"\n\n##### Task {i}"
                markdown += f"\nInstructions {task['instructions']}"

                block_path_str = ".".join(task["block_path"])
                markdown += f"\nBlock path: {block_path_str}"
                markdown += f"\nAction: {task['action']}"

        if step["action"] == "write_code":
            markdown += f"\n\n#### Code change"
            markdown += f"\n```python\n{response['change']}\n```"

            markdown += f"\n\n#### Diff"
            markdown += f"\n```diff\n{response['diff']}\n```"

    if result["test_logs"]:
        markdown += f"\n\n## Test logs"

        markdown += f"\n```\n{result['test_logs']}\n```"

    return markdown


def calculate_tokens(token_usage: list[dict]):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for token in token_usage:
        prompt_tokens += token["prompt_tokens"]
        completion_tokens += token["completion_tokens"]
        total_tokens += token["total_tokens"]

    return prompt_tokens, completion_tokens, total_tokens


def get_missing_patches(prediction: dict, golden_patch: dict) -> List[dict]:
    missing_diffs = []

    for golden_diff in golden_patch.get("diffs", []):
        diff1_covered = False
        for prediction_diff in prediction.get("diffs", []):
            if (
                golden_diff["start_line_old"] >= prediction_diff["start_line_old"]
                and golden_diff["end_line_old"] <= prediction_diff["end_line_old"]
            ):
                diff1_covered = True
                break
        if not diff1_covered:
            missing_diffs.append(golden_diff)

    return missing_diffs


def get_blocks_from_patch(codeblock: CodeBlock, patch: dict) -> List[dict]:
    spans = []
    for diff in patch.get("diffs", []):
        start_line = diff["start_line_old"]
        spans.append(Span(start_line, diff.get("end_line_old", start_line)))

    return [
        {
            "path": block.full_path() or "root",
            "block_id": block.path_string(),
            "tokens": block.sum_tokens(),
            "start_line": block.start_line,
            "end_line": block.end_line,
        }
        for block in codeblock.find_indexed_blocks_by_spans(spans)
    ]


if __name__ == "__main__":
    generate_report("pipelines/moatless_gpt-4-0125_know_files-2", overwrite=True)
