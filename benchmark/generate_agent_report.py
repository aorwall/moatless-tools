import csv
import json

from benchmark import swebench
from benchmark.utils import diff_details, diff_file_names

result_files = [
    (
        "acr-run-1",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-1/predictions_for_swebench.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-1/final_report.json",
    ),
    (
        "acr-run-2",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-2/predictions_for_swebench.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-2/final_report.json",
    ),
    (
        "acr-run-3",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-3/predictions_for_swebench.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/acr-run-3/final_report.json",
    ),
    (
        "swebench-run-1",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_1/all_preds.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_1/final_report.json",
    ),
    (
        "swebench-run-2",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_2/all_preds.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_2/final_report.json",
    ),
    (
        "swebench-run-3",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_3/all_preds.json",
        "/home/albert/repos/stuffs/auto-code-rover/results/swe-agent-results/cost_2_3/final_report.json",
    ),
]


def generate_report():

    results = {}
    run_cols = ["instance_id"]

    for result_file in result_files:
        run_name, prediction_file, final_report_file = result_file
        run_cols.append(run_name)

        with open(prediction_file, "r") as file:
            predictions = json.load(file)

        with open(final_report_file, "r") as file:
            final_report = json.load(file)

        resolved_tasks = final_report["resolved"]

        predictions_by_id = read_predictions(predictions, resolved_tasks)

        results[run_name] = {
            "resolved_tasks": resolved_tasks,
            "predictions": predictions_by_id,
        }

    csv_file = "agent_report.csv"
    with open(csv_file, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(run_cols)

    instances = swebench.get_instances(
        split="test", dataset_name="princeton-nlp/SWE-bench_Lite", data_dir="../data"
    )
    for instance in instances:
        instance_id = instance["instance_id"]
        expected_patch_file = instance["patch_files"][0]

        result_cols = [instance_id]

        for result_file in result_files:
            run_name, prediction_file, final_report_file = result_file

            if instance_id in results[run_name]["resolved_tasks"]:
                status = "resolved"
            elif (
                instance_id in results[run_name]["predictions"]
                and expected_patch_file
                in results[run_name]["predictions"][instance_id]["patch_files"]
            ):
                status = "applied"
            else:
                status = "not_found"

            result_cols.append(status)

        with open(csv_file, "a") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(result_cols)


def read_predictions(predictions: list[dict], resolved: str):
    predictions_result = {}
    for prediction in predictions:
        predictions_result[prediction["instance_id"]] = prediction

        if prediction["model_patch"]:
            predictions_result[prediction["instance_id"]]["diff_details"] = (
                diff_details(prediction["model_patch"])
            )
            predictions_result[prediction["instance_id"]]["patch_files"] = (
                diff_file_names(prediction["model_patch"])
            )
        else:
            predictions_result[prediction["instance_id"]]["diff_details"] = None
            predictions_result[prediction["instance_id"]]["patch_files"] = []

        predictions_result[prediction["instance_id"]]["resolved"] = (
            prediction["instance_id"] in resolved
        )

    return predictions_result


if __name__ == "__main__":
    generate_report()
