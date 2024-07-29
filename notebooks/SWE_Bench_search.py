# %%
import os
import sys
import json
import datetime
import litellm
import logging
import subprocess
import time
import traceback
import argparse

from moatless.index import CodeIndex, IndexSettings
from moatless.loop_search import AgenticLoop
from moatless.transitions import search_and_code_transitions, code_transitions
from moatless.workspace import Workspace
from moatless.benchmark.utils import trace_metadata
from moatless.benchmark.swebench import get_repo_dir_name, sorted_instances, setup_swebench_repo

from tqdm.notebook import tqdm
import collections

sys.path.append(os.path.abspath(os.path.join('..')))


instance_whitelist = None

# success
instance_whitelist = ["pytest-dev__pytest-5227", "django__django-16139", "sympy__sympy-24152", 
                      "django__django-16379", "django__django-16527", "django__django-13933"]

# failure
instance_whitelist_fail = [
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

# instance_whitelist = ["django__django-13933"]   # autocoderover example
# instance_whitelist = [instance_whitelist[0]]
# instance_whitelist = [instance_whitelist_fail[0]]

instance_whitelist = instance_whitelist_fail

# instance_whitelist += instance_whitelist_fail

# %%
def load_keys():
    keys_dir = "../keys.json"
    with open(keys_dir) as f:
        keys = json.load(f)
    return keys

# %%
def set_environment_variables(keys):
    for name, value in keys.items():
        os.environ[name] = value

# %%
def create_directories(evaluation_name, debug=False):
    evaluations_dir = "../evaluations"
    
    if debug:
        evaluation_name = f"{evaluation_name}_debug"
    
    evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
    trajectory_dir = f"{evaluation_dir}/trajs"
    predictions_path = f"{evaluation_dir}/all_preds.jsonl"

    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
        
    print(f"Directories created: {evaluation_dir}, {trajectory_dir}, {predictions_path}")

    return evaluation_dir, trajectory_dir, predictions_path

# %%
def evaluate(instance, evaluation_name, trajectory_dir, 
            model, temperature, max_cost, index_store_dir,
            max_actions, debug=False):
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
                       max_cost=max_cost,
                       max_actions=max_actions,
                       code_transitions=code_transitions(),
                       max_transitions=40)

    info = {
        "evaluation_name": evaluation_name,
        "instance_id": instance["instance_id"]
    }

    start_time = time.time()
    try:
        trajectory, output = loop.run_search(problem_statement)
    except Exception as e:
        info["error"] = traceback.format_exc()
        logging.exception(f"Error in evaluation of {instance['instance_id']} ")
        print(info["error"])
    finally:
        info["duration"] = time.time() - start_time
        info["total_cost"] = loop.trajectory.total_cost()

    info["submission"] = output.stdout

    trajectory.save_info(info)
    trajectory = loop.trajectory.to_dict()
    trajectory["info"] = info

    return trajectory

# %%
def run_evaluation(args, dataset="princeton-nlp/SWE-bench_Lite", split="test"):
    keys = load_keys()
    set_environment_variables(keys)

    max_cost = 0.5
    index_store_dir = "../20240522-voyage-code-2"

    evaluation_name = f"20240617_moatless_{args.model}_search_max_action_{args.max_actions}_temp_{args.temp}"
    evaluation_dir, trajectory_dir, predictions_path = create_directories(evaluation_name, 
                                                                          debug=args.debug)

    instances = sorted_instances(dataset, split)

    count = 0
    generated = 0
    error = 0
    sum_duration = 0
    sum_total_cost = 0
    failure_to_run = 0

    with open(predictions_path, "w") as file:
        file.write("")

    if instance_whitelist:
        instances = [instance for instance in instances if instance["instance_id"] in instance_whitelist]

    stats = collections.defaultdict(int)
    pbar = tqdm(instances)
    for instance in pbar:
        print(f"Instance: {instance['instance_id']}")
        try:
            trajectory = evaluate(instance, evaluation_name, 
                                trajectory_dir, args.model, 
                                args.temp, max_cost, index_store_dir,
                                max_actions=args.max_actions,
                                debug=args.debug)
        except Exception as e:
            print(f"Error in evaluation of {instance['instance_id']}")
            failure_to_run += 1

        if not trajectory:
            error += 1
            continue

        sum_duration += trajectory["info"]["duration"]
        sum_total_cost += trajectory["info"]["total_cost"]

        if trajectory["info"].get("error"):
            error += 1

        if trajectory["info"].get("submission"):
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
        
        stats['n_success'] += 1 if trajectory["info"].get("submission") else 0

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

        print(f"Instance: {instance['instance_id']} saved in {predictions_path}")

    print(f"Total instances: {count}, Generated: {generated}, Error: {error}, Failure to run: {failure_to_run}, Success: {stats['n_success']}")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation with specified temperature.")
    parser.add_argument("--temp", type=float, default=0.2, help="Temperature for the model")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--max_actions", type=int, default=1, help="Maximum number of actions to take")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    run_evaluation(args)