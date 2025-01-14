#!/usr/bin/env python3

import json
import os
from typing import Dict, List, Tuple, Set
from pathlib import Path

def load_instances() -> Set[str]:
    lite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "moatless/benchmark/swebench_lite_all_evaluations.json")
    verified_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "moatless/benchmark/swebench_verified_all_evaluations.json")

    with open(lite_path) as f:
        lite_instances = json.load(f)
    with open(verified_path) as f:
        verified_instances = json.load(f)

    lite_ids = {instance["instance_id"] for instance in lite_instances}
    verified_ids = {instance["instance_id"] for instance in verified_instances}

    all_ids = set()
    all_ids.update(lite_ids)
    all_ids.update(verified_ids)

    return all_ids

def read_resolved(eval_dir: Path) -> Set[str]:
    """Analyze a single evaluation directory and return (resolved_count, percentage)."""
    results_file = eval_dir / "results" / "results.json"
    if not results_file.exists():
        return set()
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        # Only count resolved instances that exist in the solvable dataset
        return set(instance_id for instance_id in results.get("resolved", []))
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to read {results_file}")
        return set()

def main():
    # Load solvable instances
    instance_ids = load_instances()

    lite_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "moatless/benchmark/swebench_lite_all_evaluations.json")
    verified_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "moatless/benchmark/swebench_verified_all_evaluations.json")

    with open(lite_path) as f:
        lite_instances = json.load(f)
    with open(verified_path) as f:
        verified_instances = json.load(f)

    lite_ids = {instance["instance_id"] for instance in lite_instances}
    verified_ids = {instance["instance_id"] for instance in verified_instances}

    paths = [
        "/home/albert/repos/stuffs/experiments/evaluation/test",
        "/home/albert/repos/stuffs/experiments/evaluation/lite",
        "/home/albert/repos/stuffs/experiments/evaluation/verified"
    ]

    instance_ids = sorted(instance_ids)

    resolved_by = {
    }

    for instance_id in instance_ids:
        resolved_by[instance_id] = {
            "resolved_submissions": [],
            "no_of_submissions": 0
        }

    submission_count = 0
    for base_path in paths:

        for eval_dir in Path(base_path).iterdir():
            if not eval_dir.is_dir():
                continue

            resolved_ids = read_resolved(eval_dir)

            if resolved_ids:
                submission_count += 1

            for instance_id, submission in resolved_by.items():
                if ((Path(base_path).name == "lite" and instance_id in lite_ids) or
                        (Path(base_path).name == "verified" and instance_id in verified_ids) or
                        Path(base_path).name == "test"):
                    submission["no_of_submissions"] += 1

                    if instance_id in resolved_ids:
                        submission["resolved_submissions"].append(eval_dir.name)

    for instance_id, submissions in sorted(resolved_by.items(), key=lambda x: len(x[1]["resolved_submissions"]) / x[1]["no_of_submissions"] if x[1]["no_of_submissions"] else 0):
        perentage = len(submissions["resolved_submissions"])/submissions["no_of_submissions"]  if submissions["no_of_submissions"] else 0
        print(f"{instance_id}:\t{len(submissions["resolved_submissions"])} / {submissions["no_of_submissions"]} ({perentage}%)")

    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    output_path = os.path.join(dataset_dir, "resolved_submissions.json")
    with open(output_path, "w") as f:
        json.dump(resolved_by, f, indent=2)

    print(f"{submission_count} submissions")



if __name__ == "__main__":
    main() 