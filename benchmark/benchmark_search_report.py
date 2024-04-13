import csv
import json
import os

from benchmark import swebench
from benchmark.utils import diff_details, diff_file_names


def generate_report():

    results = {}
    run_cols = ["instance_id"]

    csv_file = "agent_report.csv"
    with open(csv_file, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(run_cols)

    prompt_tokens = 0
    completion_tokens = 0

    total = 0
    instances = swebench.get_instances(
        split="test", dataset_name="princeton-nlp/SWE-bench_Lite", data_dir="../data"
    )
    for instance in instances:
        instance_id = instance["instance_id"]
        session_log_file = f"logs/search/{instance_id}/session.json"
        if not os.path.exists(session_log_file):
            print(f"Session log not found for instance {instance_id}")
            continue
        with open(session_log_file, "r") as file:
            session_log = json.load(file)

        for log in session_log:
            if "prompt_tokens" in log:
                prompt_tokens += log["prompt_tokens"]
            if "completion_tokens" in log:
                completion_tokens += log["completion_tokens"]

        total += 1

    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")

    avg_prompt_tokens = prompt_tokens / total
    avg_completion_tokens = completion_tokens / total

    print(f"Avg prompt tokens: {avg_prompt_tokens}")
    print(f"Avg completion tokens: {avg_completion_tokens}")


if __name__ == "__main__":
    generate_report()
