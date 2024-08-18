import json

import pandas as pd

from moatless.benchmark.swebench import setup_swebench_repo, sorted_instances
from moatless.benchmark.utils import get_file_spans_from_patch
from moatless.repository import FileRepository

experiments_runs = [
    "20240402_sweagent_claude3opus",
    "20240402_sweagent_gpt4",
    "20240509_amazon-q-developer-agent-20240430-dev",
    "20240523_aider",
    "20240524_opencsg_starship_gpt4",
    "20240530_autocoderover-v20240408",
    "20240604_CodeR",
    "20240612_IBM_Research_Agent101",
    "20240612_marscode-agent-dev",
    "20240612_MASAI_gpt4o",
    "20240615_appmap-navie_gpt4o",
    "20240617_factory_code_droid",
    "20240617_moatless_gpt4o",
]

experiment_verified_runs = [
    "20231010_rag_claude2",
    "20231010_rag_gpt35",
    "20231010_rag_swellama13b",
    "20231010_rag_swellama7b",
    "20240402_rag_claude3opus",
    "20240402_rag_gpt4",
    "20240402_sweagent_claude3opus",
    "20240402_sweagent_gpt4",
    "20240509_amazon-q-developer-agent-20240430-dev",
    "20240615_appmap-navie_gpt4o",
    "20240617_factory_code_droid",
    "20240620_sweagent_claude3.5sonnet",
    "20240628_autocoderover-v20240620",
    "20240721_amazon-q-developer-agent-20240719-dev",
]

# dataset_path = "/home/albert/repos/albert/moatless/datasets/swebench_lite_all_evaluations.json"
dataset_path = (
    "/home/albert/repos/albert/moatless/datasets/swebench_verified_all_evaluations.json"
)


def read_predictions(pred_path: str):
    predictions = {}
    with open(pred_path) as f:
        for line in f.readlines():
            prediction = json.loads(line)
            predictions[prediction["instance_id"]] = prediction["model_patch"]
    return predictions


def generate_report(dataset_name: str = "princeton-nlp/SWE-bench_Lite"):
    results = {}

    experiments_dir = "/home/albert/repos/stuffs/experiments/evaluation/verified"

    runs = []
    for run_name in experiment_verified_runs:
        runs.append(
            (
                run_name,
                f"{experiments_dir}/{run_name}/all_preds.jsonl",
                f"{experiments_dir}/{run_name}/results/results.json",
            )
        )

    for run_name, prediction_file, result_file in runs:
        with open(result_file) as file:
            final_report = json.load(file)

        resolved_tasks = final_report["resolved"]
        predictions_by_id = read_predictions(prediction_file)

        results[run_name] = {
            "resolved_tasks": resolved_tasks,
            "predictions": predictions_by_id,
        }

    evaluation_dataset = []

    report = []

    instances = sorted_instances(split="test", dataset_name=dataset_name)
    for instance in instances:
        instance_id = instance["instance_id"]
        expected_patch = instance["patch"]
        repo_dir = setup_swebench_repo(instance, repo_base_dir="/tmp/repos_2")
        file_repo = FileRepository(repo_dir)

        expected_file_spans = get_file_spans_from_patch(file_repo, expected_patch)

        evaluation_instance = {
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "problem_statement": instance["problem_statement"],
            "golden_patch": instance["patch"],
            "expected_spans": expected_file_spans,
            "resolved_by": [],
            "alternative_spans": [],
        }

        for run_name, _, _ in runs:
            prediction = results[run_name]["predictions"].get(instance_id)

            if instance_id not in results[run_name]["resolved_tasks"]:
                continue

            file_spans = get_file_spans_from_patch(file_repo, prediction)

            is_different = False
            alternative_spans = {}
            for file_path, span_ids in file_spans.items():
                if file_path in expected_file_spans:
                    alternative_spans[file_path] = span_ids

                    if set(expected_file_spans[file_path]).difference(set(span_ids)):
                        is_different = True

            if is_different:
                evaluation_instance["alternative_spans"].append(
                    {"run_name": run_name, "spans": alternative_spans}
                )

            resolved = {
                "name": run_name,
                "patch": prediction,
                "updated_spans": file_spans,
                "alternative_spans": alternative_spans,
            }

            evaluation_instance["resolved_by"].append(resolved)

        report.append(
            {
                "instance_id": instance_id,
                "resolved_by": len(evaluation_instance["resolved_by"]),
            }
        )

        evaluation_dataset.append(evaluation_instance)

        with open(dataset_path, "w") as f:
            json.dump(evaluation_dataset, f, indent=2)

    return pd.DataFrame(report)


if __name__ == "__main__":
    df = generate_report("princeton-nlp/SWE-bench_Verified")
