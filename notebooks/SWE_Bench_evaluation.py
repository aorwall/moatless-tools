# %% [markdown]
# # Run SWE-Bench evaluation
# This notebook describes how you can recreate the evaluation of [Moatless Tools](https://github.com/aorwall/moatless-tools) used for submission to the [SWE Bench Leaderboard](https://www.swebench.com/).

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

# %% [markdown]
# ## Download index files
# 
# Moatless Tools use a vector index to do semantic code search. To avoid having to index all repositories, you can use this shared volume with index embeddings embedded with voyage-code-2 for all instances of SWE-Bench Lite: https://drive.google.com/drive/folders/1RhG1w_VVY938ogHRhZ7K5Tapnvs5b-PW?usp=sharing
# 
# If you add a shortcut to “20240522-voyage-code-2” in "My Drive" it should be possible to mount Google Drive on `/content/drive` and find it on `/content/drive/MyDrive/20240522-voyage-code-2`

# %%
import json
keys_dir = "../keys.json"
with open(keys_dir) as f:
    keys = json.load(f)
    f.close()
index_store_dir = "../20240522-voyage-code-2"

# %% [markdown]
# To use `voyage-code-2` embeddings, you also need an API key from Voyage AI (https://www.voyageai.com/). Add this to your secrets.

# %%
from moatless.index import CodeIndex, IndexSettings
import os

os.environ["VOYAGE_API_KEY"] = keys["VOYAGE_API_KEY"]
os.environ["HUGGINGFACE_API_KEY"] = keys["HUGGINGFACE_API_KEY"]

# %% [markdown]
# ## Stage for evaluation
# Litellm is used to run requests to LLMs. Use the model names specified for Litellm and add the API Key to *Secrets*.
# 
# `model=gpt-4` with `temperature=0.2` was used in the latest subission to the SWE-Bench Lite leaderboard.
# 
# `max_cost` is set to limit how much each run is allowed to cost.

# %%
os.environ["OPENAI_API_KEY"] = keys['OPENAI_API_KEY']

model = "gpt-4o"
temperature = 0.2

max_cost=0.5

# %% [markdown]
# Enter a evaluation name and specify directories to save predictions and trajectories.

# %%
import datetime
import os

evaluations_dir = "../evaluations"
evaluation_name = f"20240617_moatless_gpt-4o-2024-05-13_demo_temp={temperature}"
evaluation_dir = f"{evaluations_dir}/{evaluation_name}"
trajectory_dir = f"{evaluations_dir}/{evaluation_name}/trajs"
predictions_path = f"{evaluation_dir}/all_preds.jsonl"

if not os.path.exists(trajectory_dir):
    os.makedirs(trajectory_dir)

print(evaluation_dir)

# %% [markdown]
# (Optional) Set up tracing. [Langfuse](https://langfuse.com/) for example .

# %%
import litellm

os.environ["LANGFUSE_PUBLIC_KEY"] = keys['LANGFUSE_PUBLIC_KEY']
os.environ["LANGFUSE_SECRET_KEY"] = keys['LANGFUSE_SECRET_KEY']

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# %% [markdown]
# Define the evaluation function.

# %%
from moatless.loop import AgenticLoop
from moatless.transitions import search_and_code_transitions
from moatless.workspace import Workspace
from moatless.benchmark.utils import trace_metadata

import json
import logging
import subprocess
import time
import traceback

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
        logging.exception(f"Error in evaluation of {instance['instance_id']} ")
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

instance_whitelist = None

# %% [markdown]
# 

# %% [markdown]
# ## Run the evaluation
# 
# Test if evaluation works with a sub set of 5 instances. Remove this to run the full benchmark.

# %%
instance_whitelist = ["pytest-dev__pytest-5227", "django__django-16139", "sympy__sympy-24152", "django__django-16379", "django__django-16527", "django__django-13933"]
# instance_whitelist = ["django__django-13933"]   # autocoderover example
instance_whitelist = [instance_whitelist[0]]
# instance_whitelist = ["django__django-13933"]

# %% [markdown]
# Run the evaluation

# %%
from moatless.benchmark.swebench import get_repo_dir_name, sorted_instances, setup_swebench_repo
from tqdm.notebook import tqdm

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


run_evaluation()


