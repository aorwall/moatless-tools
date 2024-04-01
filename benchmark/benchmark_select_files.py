import json
import logging
import os

from dotenv import load_dotenv

from benchmark.swebench import instance_reports
from benchmark.utils import recall_report
from moatless.retriever import CodeSnippet
from moatless.retrievers.code_selector import CodeSelector
from moatless.utils.repo import setup_github_repo


def get_report(instance_id: str, report_dir: str = 'reports/princeton-nlp-SWE-bench-devin'):
    report_path = os.path.join(report_dir, instance_id, "data.json")
    with open(report_path, "r") as f:
        return json.load(f)


def save_report(report, report_dir: str = 'reports/princeton-nlp-SWE-bench-devin'):
    report_path = os.path.join(report_dir, report['instance_id'], "select_file.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def benchmark_reports(report_dir: str = 'reports/princeton-nlp-SWE-bench-devin'):
    reports = instance_reports(report_dir)

    for report in reports:
        report_path = os.path.join(report_dir, report['instance_id'], "select_file.json")
        if os.path.exists(report_path):
            print(f"Report '{report_path}' already exists")
            continue

        select_files(report)


def select_files(report: dict, base_dir: str = '/tmp/repos', force: bool = False):
    print(f"Selecting files for report '{report['instance_id']}'")

    # FIXME: Just temporary workaround to clean up diffs
    patch_diff_details = {}
    for file_path, diffs in report['patch_diff_details'].items():
        cleaned_diffs = []
        patch_diff_details[file_path] = {"diffs": cleaned_diffs}
        for diff in diffs['diffs']:
            if 'file_pos' not in diff:
                print(f"Skipping {report['instance_id']} as it doesn't have all matching files anyway")
                return
            cleaned_diffs.append({
                'start_line_old': diff.get('start_line_old', 0),
                'end_line_old': diff.get('end_line_old', None)
            })
    report['patch_diff_details'] = patch_diff_details

    repo_path = setup_github_repo(repo=report['repo'], base_commit=report['base_commit'], base_dir=base_dir)

    code_selector = CodeSelector(model_name="claude-3-haiku-20240307", file_context_token_limit=100000, repo_path=repo_path)

    code_snippets = []

    for file in report['files']:
        file_path = os.path.join(repo_path, file['file_path'])
        with open(file_path, 'r') as f:
            content = f.read()

        code_snippets.append(CodeSnippet(
            id=file['file_path'],
            file_path=file['file_path'],
            content=content
        ))

    response = code_selector.select_files(report['problem_statement'], code_snippets)

    code_snippets = [selected_code_snippet.code_snippet for selected_code_snippet in response.files]

    report = recall_report(report, code_snippets, repo_path)

    report['usage'] = response.usage_stats
    report['total_snippets_listed'] = response.total_snippets_listed

    print(json.dumps(report, indent=2))
    save_report(report)


if __name__ == '__main__':
    load_dotenv('../.env')
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

#    benchmark_reports()

    report = get_report('django__django-13281')
    select_files(report)
