import csv
from moatless.benchmark.report_v2 import read_results_from_json, BenchmarkResult
from moatless.benchmark.swebench import setup_swebench_repo, create_workspace
from moatless.benchmark.utils import file_spans_to_dict, get_moatless_instance, has_identified_files, has_identified_spans
from moatless.edit import ExpandContext


def analyze_identified(instance_id: str, result: BenchmarkResult):
    instance = get_moatless_instance(instance_id, split="verified")

    expected_solutions = [instance["expected_spans"]]

    for resolved_by in instance["resolved_by"]:
        if "alternative_spans" in resolved_by and resolved_by["alternative_spans"] not in expected_solutions:
            expected_solutions.append(resolved_by["alternative_spans"])

    found_files_in_search = has_identified_files(expected_solutions, result.search.found_spans_details)
    found_spans_in_search = has_identified_spans(expected_solutions, result.search.found_spans_details)

    found_files = has_identified_files(expected_solutions, result.identify.found_spans_details)
    found_spans = has_identified_spans(expected_solutions, result.identify.found_spans_details)

    workspace = create_workspace(instance, max_file_context_tokens=4000)
    original_size = workspace.file_context.context_size()

    expand_state = ExpandContext(id=0, _workspace=workspace, _initial_message=instance["problem_statement"], expand_to_max_tokens=4000)
    expand_state.execute()

    expanded_size = workspace.file_context.context_size()

    expanded_actual_spans = file_spans_to_dict(workspace.file_context.to_files_with_spans())
    found_spans_if_expanded = has_identified_spans(expected_solutions, expanded_actual_spans)

    return {
        "instance_id": instance_id,
        "resolved_by": len(instance["resolved_by"]),
        "found_spans": found_spans,
        "found_files": found_files,
        "found_spans_in_search": found_spans_in_search,
        "found_files_in_search": found_files_in_search,
        "original_size": original_size,
        "expanded_class_size": expanded_size,
        "found_spans_if_expanded": found_spans_if_expanded,
    }


if "__main__" == __name__:
    results = read_results_from_json("/home/albert/repos/albert/swe-planner/evaluations/20240817_search_and_identify_gpt-4o/report.json")

    # results = [result for result in results if result.status != "identified" and not get_moatless_instance(result.instance_id, split="verified")["resolved_by"]]
    csv_data = []
    for i, result in enumerate(results):
        if result.instance_id != "django__django-15104":
            continue
        data = analyze_identified(result.instance_id, result)
        print(f"Analyzing {i+1}/{len(results)}: {result.instance_id}. Found files: {data['found_files']}, Found spans: {data['found_spans']}, Found spans if expanded: {data['found_spans_if_expanded']}, Found files in search: {data['found_files_in_search']}, Found spans in search: {data['found_spans_in_search']}, Original size: {data['original_size']}, Expanded class size: {data['expanded_class_size']}")
        csv_data.append(data)

        # Write results to CSV
        csv_file_path = "analysis_results_expand_state_tew000.csv"
        with open(csv_file_path, "w", newline="") as csvfile:
            fieldnames = ["instance_id", "resolved_by", "found_files", "found_spans", "found_files_in_search", "found_spans_in_search", "original_size", "expanded_class_size", "found_spans_if_expanded"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)

    print(f"Results have been written to {csv_file_path}")

#    analyze_identified("django__django-15104")