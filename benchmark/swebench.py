import json
import os

from datasets import load_dataset

from benchmark.utils import diff_details, write_json, diff_file_names


def instance_report(dir: str, subdir: str) -> dict:
    report_path = os.path.join(dir, subdir, "data.json")
    with open(report_path, "r") as f:
        return json.load(f)


def instance_reports(dir: str) -> list[dict]:
    return [instance_report(dir, subdir) for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]


def sort_key(data_row):
    text, number = data_row["instance_id"].rsplit('-', 1)
    return text, int(number)


def get_filename(split: str, dataset_name: str):
    return f'{dataset_name.replace("/", "-")}-{split}'


def download(split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Download oracle (patched files) for all rows in split"""

    file_name = get_filename(split, dataset_name)
    file_path = f"data/{file_name}.json"
    if os.path.exists(file_path):
        print(f"File '{file_path}' already exists")
        with open(file_path) as f:
            return json.load(f)

    else:
        print(f"File '{file_path}' does not exist. Downloading...")

    dataset = load_dataset(dataset_name, split=split)
    result = []
    for row_data in dataset:
        row_data["patch_files"] = diff_file_names(row_data['patch'])
        row_data["test_patch_files"] = diff_file_names(row_data['test_patch'])
        row_data["patch_diff_details"] = diff_details(row_data['patch'])
        result.append(row_data)

    file_name = get_filename(split, dataset_name)
    write_json('data', file_name, result)

    return result


def get_instance(id: str, split: str='test', dataset_name='princeton-nlp/SWE-bench'):
    dataset = download(split, dataset_name)
    for row in dataset:
        if row['instance_id'] == id:
            return row


def get_index_name(row_data: dict):
    return f"{row_data['repo'].replace('/', '_')}__{row_data['instance_id']}"