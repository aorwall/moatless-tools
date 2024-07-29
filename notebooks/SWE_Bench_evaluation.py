
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
import json

from moatless.index import CodeIndex, IndexSettings
import os
import datetime

import logging
import subprocess
import time
import traceback

from moatless.loop_feedback import AgenticLoop
from moatless.transitions import search_and_code_transitions
from moatless.workspace import Workspace
from moatless.benchmark.utils import trace_metadata

from moatless.benchmark.swebench import get_repo_dir_name, sorted_instances, setup_swebench_repo
from tqdm.notebook import tqdm


keys_dir = "../keys.json"
with open(keys_dir) as f:
    keys = json.load(f)
    f.close()
index_store_dir = "../20240522-voyage-code-2"

model = "gpt-4o"
temperature = 0.2

max_cost=0.5

evaluations_dir = "../evaluations"
evaluation_name = f"20240617_moatless_gpt-4o-debug"
evaluations_dir = f"{evaluations_dir}/{evaluation_name}"
trajectory_dir = f"{evaluations_dir}/trajs"
predictions_path = f"{evaluations_dir}/all_preds.jsonl"

os.makedirs(evaluations_dir, exist_ok=True)
os.makedirs(trajectory_dir, exist_ok=True)

def set_environment_variables(keys):
    for name, value in keys.items():
        os.environ[name] = value

set_environment_variables(keys)

instance_whitelist = None
instance_whitelist = ["pytest-dev__pytest-5227", "django__django-16139", "sympy__sympy-24152", "django__django-16379", "django__django-16527", "django__django-13933"]
INSTANCES_FAIL = [
    "django__django-11019",
    "django__django-11630",
    "django__django-12856",
    "sympy__sympy-18199",
    "sympy__sympy-19487",
    "sympy__sympy-18835",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13241",
    "matplotlib__matplotlib-24970",
    "pydata__xarray-3364",
    "sphinx-doc__sphinx-8506"
]
# instance_whitelist = ["sympy__sympy-24152"]
# instance_whitelist = ["django__django-13933"]
# instance_whitelist = [instance_whitelist[0]]
instance_whitelist = [INSTANCES_FAIL[0]]

def evaluate(instance):
    instance_id = instance["instance_id"]
    trajectory_path = os.path.join(trajectory_dir, f"{instance_id}.json")

    repo_dir = setup_swebench_repo(instance)
    persist_dir = os.path.join(
        index_store_dir, get_repo_dir_name(instance_id)
    )
    workspace = Workspace.from_dirs(repo_dir=repo_dir, index_dir=persist_dir)

    # # If you need to restart the evaluation you can read up already existing trajectories.
    # if os.path.exists(trajectory_path):
    #     with open(trajectory_path) as file:
    #         trajectory = json.load(file)
    #     if "info" in trajectory and trajectory["info"].get("submission") or "error" in trajectory["info"]:
    #         return trajectory

    problem_statement = instance["problem_statement"]

    metadata = trace_metadata(instance_id=instance_id, session_id=evaluation_name, trace_name="search_and_code")
    transitions = search_and_code_transitions(global_params={"model": model, "temperature": temperature})

    loop = AgenticLoop(transitions=transitions, 
                       workspace=workspace, 
                       metadata=metadata, 
                       trajectory_path=trajectory_path, 
                       max_cost=0.5)

    info = {
        "evaluation_name": evaluation_name,
        "instance_id": instance["instance_id"]
    }

    start_time = time.time() 
    try:
        response = loop.run(problem_statement)
    except Exception as e:
        info["error"] = traceback.format_exc()
        print(f"Error in evaluation of {instance['instance_id']}")
        logging.exception(f"Error in evaluation of {instance['instance_id']} ")
        raise e
    finally:
        info["duration"] = time.time() - start_time
        info["total_cost"] = loop.trajectory.total_cost()

    workspace.save()

    output = subprocess.run(
          ["git", "diff"],
          capture_output=True,
          text=True,
          cwd=repo_dir,
    )

    info["submission"] = output.stdout

    loop.trajectory.save_info(info)
    trajectory = loop.trajectory.to_dict()

    return trajectory

def run_evaluation(dataset: str = "princeton-nlp/SWE-bench_Lite", split="test"):
    instances = sorted_instances(dataset, split)

    count = 0
    generated = 0
    error = 0

    sum_duration = 0
    sum_total_cost = 0

    with open(predictions_path, "w") as file:
        file.write("")

    if instance_whitelist:
        instances = [instance for instance in instances if instance["instance_id"] in instance_whitelist]

    stats = {}
    pbar = tqdm(instances)
    for instance in pbar:
        print(f"Evaluating {instance['instance_id']}")

        trajectory = evaluate(instance)
        if not trajectory or "info" not in trajectory:
            error += 1
            continue

        info = trajectory.get("info", {})

        sum_duration += info.get("duration", 0)
        sum_total_cost += info.get("total_cost", 0)

        if info.get("error"):
            error += 1

        if info.get("submission"):
            generated += 1

        count += 1

        if sum_duration > 0:
            stats["avg_duration"] = sum_duration / count

        if sum_total_cost > 0:
            stats["avg_cost"] = sum_total_cost / count
            stats["total_cost"] = sum_total_cost

        if generated > 0:
            success_rate = (generated / count) * 100
            stats["generated"] = f"{success_rate:.2f}%"

        stats["error"] = error

        pbar.set_postfix(stats)

        prediction = {
            "model_name_or_path": evaluation_name,
            "instance_id": instance["instance_id"],
            "model_patch": trajectory["info"].get("submission", ""),
            "is_successful": bool(trajectory["info"].get("submission"))  # True if submission exists, False otherwise
        }

        with open(predictions_path, "a") as file:
            json_string = json.dumps(prediction)
            file.write(json_string + "\n")
        
        print(f"Prediction for {instance['instance_id']} written to {predictions_path}")


run_evaluation()


