import json
import logging
import os

from moatless.repository import FileRepository
from moatless.benchmark.swebench import (
    found_in_expected_spans,
    found_in_alternative_spans,
    setup_swebench_repo,
)
from moatless.benchmark.utils import get_missing_files
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)


def to_result(
    instance: dict, trajectory: dict, report: dict | None
) -> tuple[dict, list]:
    """
    Generate reports from saved trajectories with version 1 format.
    """

    info = trajectory["info"]

    resolved = report and info.get("instance_id", "") in report["resolved"]

    try:
        transitions = []
        result = {
            "instance_id": instance["instance_id"],
            "duration": info.get("duration", 0),
            "total_cost": info.get("total_cost", 0),
            "resolved_by": (len(instance.get("resolved_by", []))),
            "status": None,
            "transitions": len(trajectory["transitions"]),
            "edited": False,
            "planned": False,
            "identified": None,
            "expected_identified": None,
            "alt_identified": None,
            "found_in_search": None,
            "file_identified": None,
            "file_in_search": None,
            "edit_retries": 0,
            "has_diff": False,
            "lint_codes": None,
            "review": False,
            "p_query": 0,
            "p_file": 0,
            "p_code": 0,
            "p_class": 0,
            "p_function": 0,
            "lints": "",
        }

        lint_codes = set()
        search_results_spans = {}
        identified_spans = {}
        planned_spans = {}
        edited_spans = {}

        id_iterations = 0
        search_iterations = 0

        if instance.get("expected_spans"):
            for transition in trajectory["transitions"]:
                if transition["name"] not in result:
                    result[transition["name"]] = 0
                    result[f"{transition['name']}_cost"] = 0

                result[transition["name"]] += 1

                expected_span_str = ""
                for file_path, span_ids in instance["expected_spans"].items():
                    expected_span_str += f"{file_path}: {span_ids} "

                transition_result = {
                    "instance_id": instance["instance_id"],
                    "resolved": resolved,
                    "name": transition["name"],
                    "cost": 0,
                    "expected_spans": expected_span_str,
                    "actual_spans": "",
                }

                if not transition["actions"]:
                    continue

                for traj_action in transition["actions"]:
                    result[f"{transition['name']}_cost"] += traj_action.get(
                        "completion_cost", 0
                    )
                    transition_result["cost"] += traj_action.get("completion_cost", 0)

                if transition["name"] == "SearchCode":
                    search_iterations += 1

                    action = transition["actions"][-1]

                    if "search_requests" in action["action"]:
                        for search_request in action["action"]["search_requests"]:
                            if search_request.get("query"):
                                result["p_query"] += 1

                            if search_request.get("file_pattern"):
                                result["p_file"] += 1

                            if search_request.get("code_snippet"):
                                result["p_code"] += 1

                            if search_request.get("class_name") or search_request.get(
                                "class_names"
                            ):
                                result["p_class"] += 1

                            if search_request.get(
                                "function_name"
                            ) or search_request.get("function_names"):
                                result["p_function"] += 1

                    if "output" in action and action.get("output"):
                        output = action["output"]

                        if "query" in output:
                            result["p_query"] += 1

                        if "file_pattern" in output:
                            result["p_file"] += 1

                        if "code_snippet" in output:
                            result["p_code"] += 1

                        if "class_name" in output or "class_names" in output:
                            result["p_class"] += 1

                        if "function_name" in output or "function_names" in output:
                            result["p_function"] += 1

                        if output.get("ranked_spans"):
                            for ranked_span in output["ranked_spans"]:
                                if ranked_span["file_path"] not in search_results_spans:
                                    search_results_spans[ranked_span["file_path"]] = []
                                search_results_spans[ranked_span["file_path"]].append(
                                    ranked_span["span_id"]
                                )

                            if not result["found_in_search"] and (
                                found_in_expected_spans(instance, search_results_spans)
                                or found_in_alternative_spans(
                                    instance, search_results_spans
                                )
                            ):
                                result["found_in_search"] = search_iterations

                            if not result["file_in_search"]:
                                missing_files = get_missing_files(
                                    instance["expected_spans"],
                                    search_results_spans,
                                )
                                if not missing_files:
                                    result["file_in_search"] = search_iterations

                if transition["name"] == "IdentifyCode":
                    id_iterations += 1

                    action = transition["actions"][-1]
                    if action.get("action"):
                        identified_str = ""
                        if action["action"].get("identified_spans"):
                            for span in action["action"]["identified_spans"]:
                                identified_str += (
                                    f"{span['file_path']}: {span['span_ids']} "
                                )
                                if span["file_path"] not in identified_spans:
                                    identified_spans[span["file_path"]] = []

                                transition_result["actual_spans"] += (
                                    f"{span['file_path']}: {','.join(span['span_ids'])} "
                                )
                                for span_id in span["span_ids"]:
                                    identified_spans[span["file_path"]].append(span_id)
                        result["identified_spans"] = identified_str

                    if not result["file_identified"]:
                        missing_files = get_missing_files(
                            instance["expected_spans"],
                            identified_spans,
                        )
                        if not missing_files:
                            result["file_identified"] = id_iterations

                    if result[
                        "expected_identified"
                    ] is None and found_in_expected_spans(instance, identified_spans):
                        result["expected_identified"] = id_iterations

                    if result["alt_identified"] is None and found_in_alternative_spans(
                        instance, identified_spans
                    ):
                        result["alt_identified"] = id_iterations

                    if result.get("alt_identified") or result.get(
                        "expected_identified"
                    ):
                        result["identified"] = min(
                            result.get("alt_identified") or 1000,
                            result.get("expected_identified") or 1000,
                        )

                if transition["name"] == "PlanToCode":
                    action = transition["actions"][-1]["action"]
                    if action.get("action") == "review":
                        result["review"] = True

                    if "file_path" in action:
                        if "span_id" not in action:
                            logger.warning(
                                f"Span id missing in planning action in {instance['instance_id']}"
                            )
                        else:
                            file_path = action["file_path"]
                            if file_path not in planned_spans:
                                planned_spans[file_path] = []
                            planned_spans[file_path].append(action["span_id"])
                            transition_result["actual_spans"] = (
                                f"{file_path}: {action['span_id']} "
                            )

                    if not result.get("planned") and (
                        found_in_expected_spans(
                            instance,
                            planned_spans,
                        )
                        or found_in_alternative_spans(instance, planned_spans)
                    ):
                        result["planned"] = True

                if transition["name"] == "EditCode":
                    result["edit_retries"] = len(transition["actions"]) - 1

                    action = transition["actions"][-1]
                    output = action.get("output", {})

                    if output:
                        edited = output.get("diff")

                        if edited:
                            result["has_diff"] = True

                        for lint in output.get("verification_errors", []):
                            lint_codes.add(lint["code"])

                        if edited and "file_path" in transition["state"]:
                            file_path = transition["state"]["file_path"]
                            if file_path not in edited_spans:
                                edited_spans[file_path] = []
                            edited_spans[file_path].append(
                                transition["state"]["span_id"]
                            )
                            transition_result["actual_spans"] = (
                                f"{file_path}: {transition['state']['span_id']} "
                            )

                        if not result.get("edited") and (
                            found_in_expected_spans(
                                instance,
                                edited_spans,
                            )
                            or found_in_alternative_spans(instance, edited_spans)
                        ):
                            result["edited"] = True

                transitions.append(transition_result)

            if result.get("alt_identified") or result.get("expected_identified"):
                result["identified"] = min(
                    result.get("alt_identified") or 1000,
                    result.get("expected_identified") or 1000,
                )

            result["expected_files"] = list(instance["expected_spans"].keys())
            result["edited_files"] = list(edited_spans.keys())
            result["identified_spans"] = sum(
                [len(v) for v in identified_spans.values()]
            )

        result["lints"] = ",".join(lint_codes)

        if report and info.get("instance_id", "") in report["resolved"]:
            result["status"] = "resolved"
        elif result["edited"]:
            result["status"] = "edited"
        elif result["identified"]:
            result["status"] = "identified"
        elif result["found_in_search"]:
            result["status"] = "found_in_search"
        elif result["file_identified"]:
            result["status"] = "file_identified"
        else:
            result["status"] = ""

        if "error" in info:
            result["error"] = info["error"].split("\n")[0]
        else:
            result["error"] = ""

    except Exception as e:
        raise e

    return result, transitions


def generate_md_report(trajectory: dict, instance: dict):
    info = trajectory["info"]
    markdown = f"# {instance['instance_id']}\n"

    markdown += "\n## Problem statement\n"
    markdown += f"```\n{instance['problem_statement']}\n```\n"

    if "error" in trajectory["info"]:
        markdown += "\n## Error\n"
        markdown += f"```\n{trajectory['info']['error']}\n```\n"
    else:
        markdown += "\n## Prediction\n"
        markdown += f"```diff\n{info['submission']}\n```\n"

    markdown += "\n## Golden patch\n"
    markdown += f"```diff\n{instance['golden_patch']}\n```\n"

    markdown += "\n## Trajectory\n"

    repo_dir = setup_swebench_repo(instance)
    file_repo = FileRepository(repo_dir)

    for j, step in enumerate(trajectory["transitions"]):
        for i, traj_action in enumerate(step["actions"]):
            state_name = step["state"]
            markdown += f"### {j+1} {state_name} ({i+1})\n\n"

            if not traj_action.get("action"):
                continue
            action = traj_action["action"]

            if state_name == "PlanToCode":
                if action.get("scratch_pad"):
                    markdown += "*" + action["scratch_pad"] + "*"

                if action.get("instructions"):
                    markdown += f"\n\n * {action['instructions']}"

                if action.get("file_path"):
                    markdown += f"\n * {action['file_path']}"

                if action.get("span_id"):
                    markdown += f"\n * {action['span_id']}"

                if action.get("file_path") and action.get("span_id"):
                    markdown += "\n\n#### File context \n\n"
                    try:
                        file_context = FileContext(file_repo)
                        file_context.add_span_to_context(
                            action.get("file_path"),
                            action.get("span_id"),
                        )
                        markdown += file_context.create_prompt(
                            show_outcommented_code=True
                        )
                    except Exception as e:
                        logger.error(e)

            if state_name == "EditCode":
                markdown += "#### LLM Response\n\n"
                markdown += f"```\n{action.get('content', '')}\n```\n"

                output = traj_action.get("output")
                if output:
                    if output.get("diff"):
                        markdown += "#### Diff\n\n"
                        markdown += f"```diff\n{output['diff']}\n```\n"

                    if output.get("errors"):
                        markdown += "#### Errors\n\n"
                        markdown += f"{output['errors']}\n\n"

                    if output.get("message"):
                        markdown += "#### Message\n\n"
                        markdown += f"{output['message']}\n\n"

            if state_name == "ClarifyCodeChange":
                if action.get("thoughts"):
                    markdown += "*" + action["thoughts"] + "*"

                if action.get("output") and action.get("output").get("start_line"):
                    markdown += f"\n* Start Line: {action['output']['start_line']}\n"
                    markdown += f"\n* End Line: {action['output']['end_line']}\n"

            if state_name == "Finished":
                markdown += f"*{action['properties']['message']}*\n"

            if state_name == "Rejected":
                markdown += f"*{action['properties']['message']}*\n"

    markdown += "## Alternative patches\n"
    for alternative in instance["resolved_by"]:
        markdown += f"### {alternative['name']}\n"
        markdown += f"```diff\n{alternative['patch']}\n```\n"

    return markdown
