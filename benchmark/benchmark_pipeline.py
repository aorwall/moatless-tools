import json
import logging
import os
import subprocess

import litellm
from dotenv import load_dotenv

from benchmark import swebench
from benchmark.utils import get_block_paths_from_diffs
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.utils import Colors
from moatless.pipeline import CodingPipeline, PipelineState
from moatless.types import ContextFile
from moatless.utils.repo import setup_github_repo

litellm.success_callback = ["lunary"]
litellm.failure_callback = ["lunary"]

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv('../.env')


filtered_instance_ids = [
    "astropy__astropy-14365",
    "django__django-11422",
    "django__django-13448",
    "django__django-14608",
    "django__django-15252",
    "django__django-15902",
    "matplotlib__matplotlib-23987",
    "pydata__xarray-4248",
    "scikit-learn__scikit-learn-12471",
    "scikit-learn__scikit-learn-14983",
    "sympy__sympy-20049",
    "sympy__sympy-20639"
]

retry_from = "planner"

use_patch_blocks = True


base_dir = '/tmp/repos'
benchmark_name = 'moatless_gpt-4-0125_known_blocks'


def run_instance(instance_data: dict):
    print(f"{Colors.YELLOW}Running instance: {instance_data['instance_id']}{Colors.RESET}")

    path = setup_github_repo(repo=instance_data['repo'],
                             base_commit=instance_data['base_commit'],
                             base_dir=base_dir)
    print(f"{Colors.YELLOW}Cloned repo to path: {path}{Colors.RESET}")

    pipeline_dir = f"pipelines/{benchmark_name}/{instance_data['instance_id']}"
    os.makedirs(pipeline_dir, exist_ok=True)
    print(f"{Colors.YELLOW}Pipeline dir: {pipeline_dir}{Colors.RESET}")

    if use_patch_blocks:
        _use_patch_blocks(instance_data, repo_path=path, pipeline_dir=pipeline_dir)

    if retry_from:
        state_file_path = f"{pipeline_dir}/state.json"
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r') as f:
                state = PipelineState.model_validate_json(f.read())

            if retry_from == "planner":
                for task in state.tasks:
                    if task.state == "planned":
                        task.state = "rejected"

            with open(state_file_path, 'w') as f:
                f.write(state.json())

    context_files = [ContextFile(file_path=file) for file in instance_data['patch_files']]

    pipeline = CodingPipeline(path=path,
                              coding_request=instance_data['problem_statement'],
                              context_files=context_files,
                              pipeline_dir=pipeline_dir)

    pipeline.run()

    diff = subprocess.run(['git', 'diff'], cwd=path, check=True, text=True, capture_output=True)
    print(f"{Colors.YELLOW}Diff:\n{diff.stdout}{Colors.RESET}")

    prediction = {
        "model_name_or_path": benchmark_name,
        "instance_id": instance_data['instance_id'],
        "model_patch": diff.stdout
    }

    with open(f"pipelines/{benchmark_name}/predictions.jsonl", 'a') as file:
        json_string = json.dumps(prediction)
        file.write(json_string + '\n')


def _get_run_instance_ids(file_path):
    instance_ids = set()
    if not os.path.exists(file_path):
        return instance_ids

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            instance_ids.add(data['instance_id'])
    return instance_ids


def _use_patch_blocks(instance_data: dict, repo_path: str, pipeline_dir: str):
    file_path = instance_data['patch_files'][0]
    with open(os.path.join(repo_path, file_path), 'r') as f:
        file_content = f.read()

    parser = PythonParser()
    codeblock = parser.parse(file_content)

    block_paths = get_block_paths_from_diffs(codeblock, instance_data['patch_diff_details'])

    # TODO: Verify how to set lines for code outside of class/functions!

    state = PipelineState(
        coding_request=instance_data['problem_statement'],
        context_files=[ContextFile(file_path=instance_data['file_path'], block_paths=block_paths)],
    )

    state_file_path = f"{pipeline_dir}/state.json"
    with open(state_file_path, 'w') as f:
        f.write(state.json())


def run_single_instance(instance_id):
    instance_data = swebench.get_instance(instance_id, dataset_name='princeton-nlp/SWE-bench_Lite', split='dev', data_dir='../data')
    run_instance(instance_data)


def run_instances(split: str, dataset_name: str, data_dir: str):
    run_instance_ids = _get_run_instance_ids(f"pipelines/{benchmark_name}/predictions.jsonl")

    instances = swebench.get_instances(split=split, dataset_name=dataset_name, data_dir=data_dir)
    for instance_data in instances:
        if not retry_from and instance_data['instance_id'] in run_instance_ids:
            print(f"Skipping already run instance: {instance_data['instance_id']}")
            continue

        if instance_data['instance_id'] not in filtered_instance_ids:
            continue

        run_instance(instance_data)


if __name__ == '__main__':
    run_single_instance("astropy__astropy-14182")
    #run_instances(split='test', dataset_name='princeton-nlp/SWE-bench_Lite', data_dir='../data')
