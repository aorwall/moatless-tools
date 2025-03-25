import json
import argparse
import os
from typing import List, Dict, Any


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from a JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        else:
            print(f"Warning: File {file_path} does not exist. Skipping.")
            return []
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return []


def create_dataset_index(
    swegym_path: str,
    lite_path: str,
    verified_path: str,
    output_path: str,
):
    """
    Create an index file from multiple dataset files with priority handling for duplicates.

    Priority: verified > lite > swegym

    Args:
        swegym_path: Path to the SWE-Gym dataset file
        lite_path: Path to the SWE-bench Lite dataset file
        verified_path: Path to the SWE-bench Verified dataset file
        output_path: Path to save the index file
    """
    # Define dataset info
    dataset_info = {
        "swegym": {"dataset": "SWE-Gym/SWE-Gym", "split": "train"},
        "lite": {"dataset": "princeton-nlp/SWE-bench_Lite", "split": "test"},
        "verified": {"dataset": "princeton-nlp/SWE-bench_Verified", "split": "test"},
    }

    # Load datasets
    swegym_data = load_dataset(swegym_path)
    lite_data = load_dataset(lite_path)
    verified_data = load_dataset(verified_path)

    # Map instances by ID, respecting priority
    instance_map = {}

    # Process in order of lowest to highest priority
    for data, source in [(swegym_data, "swegym"), (lite_data, "lite"), (verified_data, "verified")]:
        for instance in data:
            instance_id = instance["instance_id"]

            # Create or update entry
            if instance_id not in instance_map:
                # New instance
                instance_map[instance_id] = {"instance": instance, "datasets": [dataset_info[source]], "source": source}
            else:
                # Existing instance - update source based on priority
                instance_map[instance_id]["source"] = source
                instance_map[instance_id]["instance"] = instance

                # Add this dataset to the list if not already present
                if dataset_info[source] not in instance_map[instance_id]["datasets"]:
                    instance_map[instance_id]["datasets"].append(dataset_info[source])

    # Create the index
    index = []
    for instance_id, info in instance_map.items():
        instance = info["instance"]
        index_entry = {
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "problem_statement": instance.get("problem_statement", ""),
            "resolved_count": len(instance.get("resolved_by", [])),
            "file_count": len(instance.get("expected_spans", {})),
            "datasets": info["datasets"],
            "expected_spans": instance.get("expected_spans", {}),
            "test_file_spans": instance.get("test_file_spans", {}),
            "resolved_by": instance.get("resolved_by", []),
        }
        index.append(index_entry)

    # Write the index to file
    with open(output_path, "w") as f:
        json.dump({"instances": index}, f, indent=2)

    print(f"Index created with {len(index)} instances and saved to {output_path}")
    print(f"Sources: {len(swegym_data)} from SWE-Gym, {len(lite_data)} from Lite, {len(verified_data)} from Verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an index from multiple dataset files")
    parser.add_argument(
        "--swegym_path", type=str, default="swebench_swegym_evaluations.json", help="Path to the SWE-Gym dataset file"
    )
    parser.add_argument(
        "--lite_path",
        type=str,
        default="moatless/evaluation/swebench_lite_all_evaluations.json",
        help="Path to the SWE-bench Lite dataset file",
    )
    parser.add_argument(
        "--verified_path",
        type=str,
        default="moatless/evaluation/swebench_verified_all_evaluations.json",
        help="Path to the SWE-bench Verified dataset file",
    )
    parser.add_argument("--output_path", type=str, default="dataset_index.json", help="Path to save the index file")

    args = parser.parse_args()

    create_dataset_index(
        swegym_path=args.swegym_path,
        lite_path=args.lite_path,
        verified_path=args.verified_path,
        output_path=args.output_path,
    )
