import os

from benchmark import swebench
from moatless.codeblocks.parser.python import PythonParser
from moatless.coder import Coder, do_diff
from moatless.pipeline import PipelineState
from moatless.utils.repo import setup_github_repo
from tests.test_merge_codeblocks import find_by_path_recursive


def _verify_write_code(original_code, update, block_path):
    coder = Coder(repo_path=None)

    parser = PythonParser()
    original_codeblock = parser.parse(original_code)
    print("Orignal")
    print(original_codeblock.to_tree())

    update_block = parser.parse(update)
    print("Update")
    print(update_block.to_tree())

    changed_block = find_by_path_recursive(original_codeblock, block_path)
    print("To be changed block")
    print(changed_block.to_tree())

    result = coder._write_code(original_codeblock, block_path, update)

    diff = do_diff("file", original_code, result.content)
    print(diff)


def create_regression_test(
    benchmark_name: str, instance_id: str, base_dir: str = "/tmp/repos"
):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )

    path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )

    with open(
        f"pipelines/{benchmark_name}/{instance_data['instance_id']}/state.json", "r"
    ) as f:
        pipeline_state = PipelineState.model_validate_json(f.read())

    for step in pipeline_state.steps:
        task = pipeline_state.tasks[0]

        if step.action == "write_code":
            change = step.response["change"]

            with open(os.path.join(path, step.response["file_path"]), "r") as f:
                original_code = f.read()

            coder = Coder(repo_path=None)

            parser = PythonParser()
            original_codeblock = parser.parse(original_code)

            result = coder._write_code(original_codeblock, task.block_path, change)
            diff = do_diff("file", original_code, result.content)

            os.makedirs(
                f"../tests/data/python/regressions/{instance_id}", exist_ok=True
            )

            with open(
                f"../tests/data/python/regressions/{instance_id}/original.py", "w"
            ) as f:
                f.write(original_code)

            with open(
                f"../tests/data/python/regressions/{instance_id}/expected.py", "w"
            ) as f:
                f.write(result.content)

            with open(
                f"../tests/data/python/regressions/{instance_id}/update.py", "w"
            ) as f:
                f.write(change)

            with open(
                f"../tests/data/python/regressions/{instance_id}/diff.txt", "w"
            ) as f:
                f.write(diff)

            with open(
                f"../tests/data/python/regressions/{instance_id}/block_path.txt", "w"
            ) as f:
                f.write(".".join(task.block_path))


def run_test_case(benchmark_name: str, instance_id: str, base_dir: str = "/tmp/repos"):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )

    path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )

    with open(
        f"pipelines/{benchmark_name}/{instance_data['instance_id']}/state.json", "r"
    ) as f:
        pipeline_state = PipelineState.model_validate_json(f.read())

    for step in pipeline_state.steps:
        task = pipeline_state.tasks[0]

        if step.action == "write_code":
            change = step.response["change"]

            with open(os.path.join(path, step.response["file_path"]), "r") as f:
                original_code = f.read()

            _verify_write_code(original_code, change, task.block_path)


create_regression_test("moatless_gpt-4-0125_know_files", "sympy__sympy-18698")
