import asyncio
import json
import logging
import os
import shutil
import traceback
import pandas as pd

from moatless.benchmark.swebench import sorted_instances
from moatless.benchmark.swebench.utils import create_repository_async
from moatless.evaluation.utils import get_file_spans_from_patch

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def read_predictions(pred_path: str):
    predictions = {}

    try:
        if not os.path.exists(pred_path):
            print(f"Missing {pred_path}")
            return predictions
        with open(pred_path) as f:
            content = f.read()
            all_preds = []
            try:
                all_preds = json.loads(content)
            except json.JSONDecodeError:
                try:
                    for line in content.split("\n"):
                        all_preds.append(json.loads(line))
                except Exception:
                    logging.exception(f"Error reading predictions from {pred_path}")

            for prediction in all_preds:
                predictions[prediction["instance_id"]] = prediction["model_patch"]

    except Exception as e:
        print(f"Error reading predictions from {pred_path}: {e}")
        raise e
    return predictions


def create_dataset(
    dataset_path: str,
    experiments_dir: str | None = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    results = {}

    runs = []
    if experiments_dir:
        for run_name in os.listdir(experiments_dir):
            if not os.path.exists(f"{experiments_dir}/{run_name}/all_preds.jsonl"):
                print(f"Missing {experiments_dir}/{run_name}/all_preds.jsonl")
                continue

            runs.append(
                (
                    run_name,
                    f"{experiments_dir}/{run_name}/all_preds.jsonl",
                    f"{experiments_dir}/{run_name}/results/results.json",
                )
            )

        print(f"Found {len(runs)} runs")

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

    if os.path.exists(dataset_path):
        with open(dataset_path) as f:
            evaluation_dataset = json.load(f)

    report = []

    instances = sorted_instances(split=split, dataset_name=dataset_name)
    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        if any(item["instance_id"] == instance_id for item in evaluation_dataset):
            print(f"Skipping {instance_id} ({i}/{len(instances)})")
            continue

        print(f"Processing {instance_id} ({i}/{len(instances)})")
        file_repo = None
        try:
            file_repo = asyncio.run(create_repository_async(instance))
            expected_file_spans = get_file_spans_from_patch(file_repo, instance["patch"])
            test_file_spans = get_file_spans_from_patch(file_repo, instance["test_patch"])

            evaluation_instance = {
                "instance_id": instance_id,
                "repo": instance["repo"],
                "base_commit": instance["base_commit"],
                "problem_statement": instance["problem_statement"],
                "golden_patch": instance["patch"],
                "test_patch": instance["test_patch"],
                "fail_to_pass": instance["FAIL_TO_PASS"],
                "pass_to_pass": instance["PASS_TO_PASS"],
                "expected_spans": expected_file_spans,
                "test_file_spans": test_file_spans,
                "resolved_by": [],
                "alternative_spans": [],
            }

            max_files = len(expected_file_spans.keys())
            min_files = len(expected_file_spans.keys())

            for run_name, _, _ in runs:
                prediction = results[run_name]["predictions"].get(instance_id)

                if instance_id not in results[run_name]["resolved_tasks"]:
                    continue

                file_spans = get_file_spans_from_patch(file_repo, prediction)

                if len(file_spans.keys()) > max_files:
                    max_files = len(file_spans.keys())

                if len(file_spans.keys()) < min_files:
                    min_files = len(file_spans.keys())

                is_different = False
                alternative_spans = {}
                for file_path, span_ids in file_spans.items():
                    if file_path in expected_file_spans:
                        alternative_spans[file_path] = span_ids

                        if set(expected_file_spans[file_path]).difference(set(span_ids)):
                            is_different = True

                if is_different:
                    evaluation_instance["alternative_spans"].append({"run_name": run_name, "spans": alternative_spans})

                resolved = {
                    "name": run_name,
                    "updated_spans": file_spans,
                    "alternative_spans": alternative_spans,
                }

                evaluation_instance["resolved_by"].append(resolved)

            report.append(
                {
                    "instance_id": instance_id,
                    "resolved_by": len(evaluation_instance["resolved_by"]),
                    "max_files": max_files,
                    "min_files": min_files,
                    "expected_files": len(expected_file_spans.keys()),
                }
            )

            evaluation_dataset.append(evaluation_instance)

            with open(dataset_path, "w") as f:
                json.dump(evaluation_dataset, f, indent=2)
        except Exception as e:
            print(f"Error processing {instance_id}")
            print(traceback.format_exc())
            # write failed instance to file
            with open("failed_instances.jsonl", "a") as f:
                json.dump({"instance_id": instance_id, "error": str(e)}, f)
                f.write("\n")
        finally:
            if file_repo:
                shutil.rmtree(file_repo.repo_dir)

    return pd.DataFrame(report)


if __name__ == "__main__":
    dataset_path = "swebench_swegym_evaluations.json"
    df = create_dataset(
        dataset_path,
        # experiments_dir="/home/albert/repos/stuffs/experiments/evaluation/lite",
        dataset_name="SWE-Gym/SWE-Gym",
        split="train",
    )

    # df to csv
    df.to_csv("evaluation_dataset.csv", index=False)
