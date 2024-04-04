import json
import logging
import os
import subprocess

import litellm
from dotenv import load_dotenv

from benchmark import swebench
from moatless.codeblocks.utils import Colors
from moatless.pipeline import CodingPipeline
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


base_dir = '/tmp/repos'
benchmark_name = 'moatless_gpt-4-0125_know_files'


def run_instance(instance_data: dict):
    print(f"{Colors.YELLOW}Running instance: {instance_data['instance_id']}{Colors.RESET}")

    path = setup_github_repo(repo=instance_data['repo'],
                             base_commit=instance_data['base_commit'],
                             base_dir=base_dir)
    print(f"{Colors.YELLOW}Cloned repo to path: {path}{Colors.RESET}")

    pipeline_dir = f"pipelines/{benchmark_name}/{instance_data['instance_id']}"
    os.makedirs(pipeline_dir, exist_ok=True)
    print(f"{Colors.YELLOW}Pipeline dir: {pipeline_dir}{Colors.RESET}")

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


def run_single_instance(instance_id):
    instance_data = swebench.get_instance(instance_id, dataset_name='princeton-nlp/SWE-bench_Lite', split='dev', data_dir='../data')
    run_instance(instance_data)


def run_instances(split: str, dataset_name: str, data_dir: str):
    run_instance_ids = _get_run_instance_ids(f"pipelines/{benchmark_name}/predictions.jsonl")

    instances = swebench.get_instances(split=split, dataset_name=dataset_name, data_dir=data_dir)
    for instance_data in instances:
        if instance_data['instance_id'] in run_instance_ids:
            print(f"Skipping instance: {instance_data['instance_id']}")
            continue
        run_instance(instance_data)


if __name__ == '__main__':
    #run_single_instance("marshmallow-code__marshmallow-1343")
    run_instances(split='test', dataset_name='princeton-nlp/SWE-bench_Lite', data_dir='../data')
