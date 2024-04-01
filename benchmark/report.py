import csv
import json
import os


def instance_report(dir: str, file: str = "data.json") -> dict:
    report_path = os.path.join(dir, file)
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r") as f:
        return json.load(f)


def instance_reports(dir: str, file: str = "data.json") -> list[str]:
    return [os.path.join(dir, subdir) for subdir in os.listdir(dir) if os.path.exists(os.path.join(dir, subdir, file))]


def sort_key(data_row):
    text, number = data_row["instance_id"].rsplit('-', 1)
    return text, int(number)


def generate_summary(dir: str):
    report_dirs = instance_reports(dir)

    summary = []

    for report_dir in report_dirs:
        report = instance_report(report_dir)

        select_report = instance_report(report_dir, "select_file.json")

        any_file_context = None
        any_snippet_context = None
        all_file_context = 0
        all_snippet_context = 0

        any_files_selected_context = None
        all_files_selected_context = 0

        avg_distance_to_snippet = []

        patch_diffs = 0

        missed_by_devin = len(set(report["patch_files"]) - set(report["model_patch_files"]))

        for file_path, file_diff in report['patch_diff_details'].items():

            for diff in file_diff['diffs']:
                patch_diffs += 1

                # Ignore new files
                if diff.get('start_line_old', 0) == 0 and diff.get('end_line_old', 0) == 0:
                    continue

                if 'file_context_length' in diff:
                    any_file_context = min(any_file_context or float('inf'), diff['file_context_length'])

                    if all_file_context is not None:
                        all_file_context = max(all_file_context, diff['file_context_length'])
                else:
                    all_file_context = None

                if 'context_length' in diff:
                    any_snippet_context = min(any_snippet_context or float('inf'), diff['context_length'])

                    if all_snippet_context is not None:
                        all_snippet_context = max(all_snippet_context, diff['context_length'])
                else:
                    if 'closest_snippet_line_distance' in diff:
                        avg_distance_to_snippet.append(diff['closest_snippet_line_distance'])
                    all_snippet_context = None

        if select_report:
            for file_path, file_diff in select_report['patch_diff_details'].items():
                for diff in file_diff['diffs']:
                    if 'file_context_length' in diff:
                        any_files_selected_context = min(any_files_selected_context or float('inf'), diff['file_context_length'])

                        if all_files_selected_context is not None:
                            all_files_selected_context = max(all_files_selected_context, diff['file_context_length'])
                    else:
                        all_files_selected_context = None

        avg_distance_to_snippet = sum(avg_distance_to_snippet) / len(avg_distance_to_snippet) if avg_distance_to_snippet else None

        summary.append({
            'instance_id': report['instance_id'],
            'vectors': report['vectors'],
            'patch_files': len(report['patch_diff_details']),
            'patch_diffs': patch_diffs,
            'any_file_context': any_file_context,
            'any_snippet_context': any_snippet_context,
            'all_file_context': all_file_context,
            'all_snippet_context': all_snippet_context,
            'avg_distance_to_snippet': avg_distance_to_snippet,
            'has_select': select_report is not None,
            'all_files_selected_context': all_files_selected_context,
            'any_files_selected_context': any_files_selected_context,
            'files_missed_by_devin': missed_by_devin,
            'devin_pass_or_fail': report['pass_or_fail']
        })

    summary.sort(key=sort_key)
    json.dump(summary, open(os.path.join(dir, "summary.json"), "w"), indent=2)

    return summary

def generate_markdown(
        dir: str,
        dataset_name: str,
        embedding_model: str,
        splitter: str,
        summary: list[dict]):
    thresholds = [13000, 27000, 50000, 200000]
    count_keys = ['all_snippet_context', 'any_snippet_context', 'all_file_context', 'any_file_context']
    counts = {key: [0] * len(thresholds) for key in count_keys}

    def update_counts(instance_report: dict):
        for i, threshold in enumerate(thresholds):
            for key in count_keys:
                if key not in instance_report:
                    continue

                if instance_report[key] is None:
                    continue

                if instance_report[key] < threshold:
                    counts[key][i] += 1

    def avg_percent(count_key: str, idx: int):
        count = counts[count_key][idx]
        return round((count / len(summary)) * 100, 2)

    md_table = "| Instance ID | Vectors | Patch Files | Any File Context | Any Snippet Context | All File Context | All Snippet Context | Avg Distance to Snippet |\n"
    md_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"

    for instance_report in summary:
        update_counts(instance_report)
        instance_id = instance_report['instance_id']

        avg_distance_to_snippet = int(instance_report['avg_distance_to_snippet']) if instance_report['avg_distance_to_snippet'] else '-'
        md_table += (f"| [{instance_id}]({instance_id}/report.md) "
                     f"| {instance_report['vectors']} "
                     f"| {instance_report['patch_files']} "
                     f"| {instance_report['any_file_context'] or '-'} "
                     f"| {instance_report['any_snippet_context'] or '-'} "
                     f"| {instance_report['all_file_context'] or '-'} "
                     f"| {instance_report['all_snippet_context'] or '-'} "
                     f"| {avg_distance_to_snippet} |\n")

    md = "# Benchmark summary\n\n"
    md += f"* **Dataset:** {dataset_name}\n"
    md += f"* **Embedding model:** {embedding_model}\n"
    md += f"* **Splitter:** `{splitter}` \n\n"

    def recall(scope: str):
        recall_md = "| | 13k | 27k | 50k | 200k |\n"
        recall_md += "| --- | --- | --- | --- | ---- |\n"
        count_key = f"all_{scope}_context"
        recall_md += f"| All | {avg_percent(count_key, 0)}% | {avg_percent(count_key, 1)}% | {avg_percent(count_key, 2)}% | {avg_percent(count_key, 3)}% |\n"
        count_key = f"any_{scope}_context"
        recall_md += f"| Any | {avg_percent(count_key, 0)}% | {avg_percent(count_key, 1)}% | {avg_percent(count_key, 2)}% | {avg_percent(count_key, 3)}% |\n"
        return recall_md

    md += "## Recall\n\n"
    md += "### File recall\n\n"
    md += recall("file")

    md += "\n### Snippet recall\n\n"
    md += recall("snippet")

    md += "\n## Instances\n\n"
    md += md_table

    with open(f"{dir}/README.md", "w") as f:
        f.write(md)

def generate_csv(summary: list[dict]):
    with open('summary.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(summary[0].keys())
        for row in summary:
            writer.writerow(row.values())


if __name__ == "__main__":
    summary = generate_summary("reports/princeton-nlp-SWE-bench-devin")
    generate_markdown(
        "reports/princeton-nlp-SWE-bench-devin",
        "princeton-nlp/SWE-bench_Lite",
        "text-embedding-3-small",
        "EpicSplitter(chunk_size=750, min_chunk_size=100, comment_strategy=CommentStrategy.ASSOCIATE",
        summary)

    generate_csv(summary)
