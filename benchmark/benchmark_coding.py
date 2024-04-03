import json
import os
import subprocess
from typing import List

from dotenv import load_dotenv

from benchmark import swebench
from moatless.codeblocks.utils import Colors
from moatless.coder import Coder
from moatless.planner import Planner, DevelopmentTask
from moatless.types import BlockPath
from moatless.utils.repo import setup_github_repo

import logging

load_dotenv('../.env')

logging.basicConfig(level=logging.INFO)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BenchmarkWorkspace:

    def __init__(self,
                 instance_data: dict,
                 base_dir: str = '/tmp/repos',
                 benchmark_id: str = None):
        self._instance_data = instance_data
        self._path = setup_github_repo(repo=instance_data['repo'], base_commit=instance_data['base_commit'], base_dir=base_dir)
        print(f"{Colors.YELLOW}Cloned repo to path: {self._path}{Colors.RESET}")
        self._benchmark_id = benchmark_id
        self._dev_planner = Planner(repo_path=self._path, model_name='gpt-4-0125-preview')
        self._coder = Coder(repo_path=self._path, model_name='gpt-4-0125-preview', log_file=f"logs/coder_{benchmark_id}_{instance_data['instance_id']}.log")

    def code(self, task):
        print(f"{Colors.YELLOW}Updating code in {task.block_path} in {task.file_path}...{Colors.RESET}")
        code_response = self._coder.write_code(main_objective=self._instance_data['problem_statement'],
                                               instructions=task.instructions,
                                               file_path=task.file_path,
                                               block_path=task.block_path)

        print(f"{Colors.GREEN}Updated code:")
        print(f"{code_response.change}{Colors.RESET}")

        if code_response.error:
            print(f"{Colors.GREEN}\nError: {code_response.error}{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}\nDiff:\n{code_response.diff}{Colors.RESET}")

        print(f"{Colors.YELLOW}Usage stats: {code_response.usage_stats}{Colors.RESET}")

        return code_response

    def plan(self, block_paths: List[BlockPath] = None):
        print(f"{Colors.YELLOW}Planning development...{Colors.RESET}")

        plan_response = self._dev_planner.plan_development(
            instructions=self._instance_data['problem_statement'],
            files=self._instance_data['patch_files'],
            block_paths=block_paths)
        print(f"{Colors.GREEN}Thoughts: {plan_response.thoughts}\n")

        for i, task in enumerate(plan_response.tasks):
            print(f"\n{Colors.GREEN}Task {i + 1}")
            print(f"Instructions: {task.instructions}")
            print(f"File path: {task.file_path}, Block path: {task.block_path}{Colors.RESET}")

        print(f"{Colors.YELLOW}Usage stats: {plan_response.usage_stats}{Colors.RESET}")

        return plan_response

    def run(self, block_paths: List[BlockPath] = None):
        print(f"{Colors.BLUE}Problem statement: {self._instance_data['problem_statement']}")
        print(f"Expected patch: {self._instance_data['patch']}{Colors.RESET}")

        result = []

        plan_response = self.plan(block_paths=block_paths)
        result.append(plan_response.dict())
        for task in plan_response.tasks:
            #input(f"{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")
            code_response = self.code(task)
            result.append(code_response.dict())

        print(f"{Colors.YELLOW}Finished with diff:{Colors.RESET}")

        diff = subprocess.run(['git', 'diff'], cwd=self._path, check=True, text=True, capture_output=True)
        print(f"{Colors.YELLOW}{diff.stdout}{Colors.RESET}")

        prediction = {
            "model_name_or_path": self._benchmark_id,
            "instance_id": self._instance_data['instance_id'],
            "model_patch": diff.stdout,
        }

        with open(f"{self._benchmark_id}_preds.jsonl", 'a') as file:
            json_string = json.dumps(prediction)
            file.write(json_string + '\n')

        return result


def run_instance(instance_id):
    instance_data = swebench.get_instance(instance_id, dataset_name='princeton-nlp/SWE-bench_Lite', split='dev', data_dir='../data')
    workspace = BenchmarkWorkspace(instance_data)
    workspace.run()


def run_task(instance_id):
    instance_data = swebench.get_instance(instance_id, dataset_name='princeton-nlp/SWE-bench_Lite', split='dev', data_dir='../data')
    workspace = BenchmarkWorkspace(instance_data)
    workspace.code(DevelopmentTask(file_path='src/sqlfluff/rules/L060.py',
                                   block_path='Rule_L060._eval',
                                   instructions="Modify the `LintResult` call within the `_eval` method to include a dynamic error message. Capture the `context.segment.raw_upper` value and use it to construct a specific error message indicating the function that should be replaced. For example, if `context.segment.raw_upper` is 'IFNULL', the error message should be 'Use 'COALESCE' instead of 'IFNULL'.'. Similarly, if it is 'NVL', the message should be 'Use 'COALESCE' instead of 'NVL'.'. Use this dynamic message when creating the `LintResult` instance."))


def run_instances(test_run: str = None, select_code_results: str = None):
    if test_run:
        print(f"{Colors.YELLOW}Running test run: {test_run}{Colors.RESET}")
        if os.path.exists(f"test_run_{test_run}.json"):
            with open(f"test_run_{test_run}.json", 'r') as f:
                test_run_data = json.load(f)
        else:
            test_run_data = {}

    if select_code_results:
        with open(select_code_results, 'r') as f:
            select_code_data = json.load(f)
    else:
        select_code_data = {}

    instances = swebench.get_instances(dataset_name='princeton-nlp/SWE-bench_Lite', split='dev', data_dir='../data')
    for instance_data in instances:
        if test_run and instance_data['instance_id'] in test_run_data:
            print(f"{Colors.YELLOW}Skipping {instance_data['instance_id']}{Colors.RESET}")
            continue

        workspace = BenchmarkWorkspace(instance_data, benchmark_id=test_run)
        #input(f"{Colors.YELLOW}Press Enter to run {instance_data['instance_id']}...{Colors.RESET}")

        if instance_data['instance_id'] in select_code_data:
            block_paths = select_code_data[instance_data['instance_id']]['block_paths']
        else:
            block_paths = []

        result = workspace.run(block_paths=block_paths)

        if test_run:
            test_run_data[instance_data['instance_id']] = result
            with open(f"test_run_{test_run}.json", 'w') as f:
                json.dump(test_run_data, f)


if __name__ == '__main__':
    run_instances("moatless_gpt-4-0125_only_patch_block_paths", select_code_results="benchmark_select_test.json")
