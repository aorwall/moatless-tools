import logging

from moatless import FileRepository
from moatless.benchmark.swebench import found_in_expected_spans, found_in_alternative_spans, setup_swebench_repo
from moatless.benchmark.utils import get_missing_files
from moatless.edit.plan import ApplyChange
from moatless.file_context import FileContext
from moatless.find.search import SearchRequest

logger = logging.getLogger(__name__)

import logging

from moatless import FileRepository
from moatless.benchmark.swebench import found_in_expected_spans, found_in_alternative_spans, setup_swebench_repo
from moatless.benchmark.utils import get_missing_files
from moatless.file_context import FileContext

logger = logging.getLogger(__name__)

import logging
from typing import Dict, List, Tuple, Optional

from moatless import FileRepository
from moatless.benchmark.swebench import found_in_expected_spans, found_in_alternative_spans, setup_swebench_repo
from moatless.benchmark.utils import get_missing_files
from moatless.file_context import FileContext
from moatless.trajectory import Trajectory
from moatless.schema import ActionTransaction, Usage, Content
from moatless.state import AgenticState

logger = logging.getLogger(__name__)


def to_result(instance: Dict, trajectory: Trajectory, report: Optional[Dict] = None) -> Dict:
    info = trajectory._info

    if report and "resolved_ids" in report and instance["instance_id"] in report["resolved_ids"]:
        result_status = "resolved"
    else:
        result_status = info.get("status")

    resolved = result_status == "resolved"

    try:
        result = {
            "instance_id": instance["instance_id"],
            "duration": info.get("duration", 0),
            "total_cost": info.get("total_cost", 0),
            "resolved_by": (len(instance.get("resolved_by", []))),
            "status": None,
            "result_status": result_status,
            "transitions": len(trajectory.transitions),
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
        search_results_spans: Dict[str, List[str]] = {}
        identified_spans: Dict[str, List[str]] = {}
        planned_spans: Dict[str, List[str]] = {}
        edited_spans: Dict[str, List[str]] = {}

        id_iterations = 0
        search_iterations = 0

        selected_transition_ids = []
        current_state = trajectory.get_current_state()
        while current_state:
            selected_transition_ids.append(current_state.id)
            current_state = current_state.previous_state

        logger.info(f"Selected transitions: {selected_transition_ids}")

        if instance.get("expected_spans"):
            for transition in trajectory.transitions:
                if selected_transition_ids and transition.id not in selected_transition_ids:
                    continue

                state: AgenticState = transition.state
                state_name = state.name

                if state_name not in result:
                    result[state_name] = 0
                    result[f"{state_name}_cost"] = 0

                result[state_name] += 1

                expected_span_str = ""
                for file_path, span_ids in instance["expected_spans"].items():
                    expected_span_str += f"{file_path}: {span_ids} "

                if not state._actions:
                    continue

                for action in state._actions:
                    result[f"{state_name}_cost"] += action.usage.completion_cost if action.usage else 0

                if state_name == "SearchCode":
                    search_iterations += 1

                    action = state._actions[-1]

                    if isinstance(action.request, SearchRequest):
                        for search_request in action.request.search_requests:
                            if search_request.query:
                                result["p_query"] += 1
                            if search_request.file_pattern:
                                result["p_file"] += 1
                            if search_request.code_snippet:
                                result["p_code"] += 1
                            if search_request.class_name or search_request.class_names:
                                result["p_class"] += 1
                            if search_request.function_name or search_request.function_names:
                                result["p_function"] += 1

                if state_name == "IdentifyCode":
                    id_iterations += 1

                    if state.ranked_spans:
                        for ranked_span in state.ranked_spans:
                            if ranked_span.file_path not in search_results_spans:
                                search_results_spans[ranked_span.file_path] = []
                            search_results_spans[ranked_span.file_path].append(ranked_span.span_id)

                        if not result["found_in_search"] and (
                                found_in_expected_spans(instance, search_results_spans)
                                or found_in_alternative_spans(instance, search_results_spans)
                        ):
                            result["found_in_search"] = search_iterations

                        if not result["file_in_search"]:
                            missing_files = get_missing_files(
                                instance["expected_spans"],
                                search_results_spans,
                            )
                            if not missing_files:
                                result["file_in_search"] = search_iterations

                    if state._actions:
                        action = state._actions[-1]
                        identified_str = ""
                        if action.request.identified_spans:
                            for span in action.request.identified_spans:
                                identified_str += f"{span.file_path}: {span.span_ids} "
                                if span.file_path not in identified_spans:
                                    identified_spans[span.file_path] = []

                                for span_id in span.span_ids:
                                    identified_spans[span.file_path].append(span_id)
                        result["identified_spans"] = identified_str

                    if not result["file_identified"]:
                        missing_files = get_missing_files(
                            instance["expected_spans"],
                            identified_spans,
                        )
                        if not missing_files:
                            result["file_identified"] = id_iterations

                    if result["expected_identified"] is None and found_in_expected_spans(instance, identified_spans):
                        result["expected_identified"] = id_iterations

                    if result["alt_identified"] is None and found_in_alternative_spans(instance, identified_spans):
                        result["alt_identified"] = id_iterations

                    if result.get("alt_identified") or result.get("expected_identified"):
                        result["identified"] = min(
                            result.get("alt_identified") or 1000,
                            result.get("expected_identified") or 1000,
                        )

                if state_name == "PlanToCode":
                    action = state._actions[-1]

                    if action.request.action == "review":
                        result["review"] = True

                    if action.request.file_path:
                        file_path = action.request.file_path
                        if file_path not in planned_spans:
                            planned_spans[file_path] = []
                        planned_spans[file_path].append(action.request.span_id)

                    if not result.get("planned") and (
                            found_in_expected_spans(instance, planned_spans)
                            or found_in_alternative_spans(instance, planned_spans)
                    ):
                        result["planned"] = True

                if state_name == "EditCode":
                    result["edit_retries"] = len(state._actions) - 1

                    action = state._actions[-1]
                    edited = action.response and action.response.trigger == "finish"

                    if edited and hasattr(state, 'file_path'):
                        file_path = state.file_path
                        if file_path not in edited_spans:
                            edited_spans[file_path] = []
                        edited_spans[file_path].append(state.span_id)

                    if not result.get("edited") and (
                            found_in_expected_spans(instance, edited_spans)
                            or found_in_alternative_spans(instance, edited_spans)
                    ):
                        result["edited"] = True

                    if action.response and action.response.output:
                        output = action.response.output
                        if edited:
                            result["has_diff"] = True

                        for lint in output.get("verification_errors", []):
                            lint_codes.add(lint["code"])

            if result.get("alt_identified") or result.get("expected_identified"):
                result["identified"] = min(
                    result.get("alt_identified") or 1000,
                    result.get("expected_identified") or 1000,
                )

            result["expected_files"] = list(instance["expected_spans"].keys())
            result["edited_files"] = list(edited_spans.keys())
            result["identified_spans"] = sum(len(v) for v in identified_spans.values())

        result["lints"] = ",".join(lint_codes)

        if result["edited"]:
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

    return result

def generate_md_report(trajectory: Trajectory, instance: Dict) -> str:
    info = trajectory._info
    markdown = f"# {instance['instance_id']}\n"

    markdown += "\n## Problem statement\n"
    markdown += f"```\n{instance['problem_statement']}\n```\n"

    if "error" in trajectory._info:
        markdown += "\n## Error\n"
        markdown += f"```\n{trajectory._info['error']}\n```\n"
    else:
        markdown += "\n## Prediction\n"
        markdown += f"```diff\n{info['submission']}\n```\n"

    markdown += "\n## Golden patch\n"
    markdown += f"```diff\n{instance['golden_patch']}\n```\n"

    markdown += "\n## Trajectory\n"

    repo_dir = setup_swebench_repo(instance)
    file_repo = FileRepository(repo_dir)

    for j, transition in enumerate(trajectory.transitions):
        state = transition.state
        for i, action in enumerate(state._actions):
            markdown += f"### {j+1} {state.name} ({i+1})\n\n"

            if state.name == "PlanToCode":
                if action.request.file_path:
                    if action.request.instructions:
                        markdown += f"\n\n * {action.request.instructions}"
                    markdown += f"\n * {action.request.file_path}"
                    markdown += f"\n * {action.request.span_id}"

                    markdown += "\n\n#### File context \n\n"
                    try:
                        file_context = FileContext(file_repo)
                        file_context.add_span_to_context(
                            action.request.file_path,
                            action.request.span_id,
                        )
                        markdown += file_context.create_prompt(
                            show_outcommented_code=True
                        )
                    except Exception as e:
                        logger.error(e)

            if state.name == "EditCode":
                markdown += "#### LLM Response\n\n"
                markdown += f"```\n{action.request.content if isinstance(action.request, Content) else ''}\n```\n"

                if action.response and action.response.output:
                    output = action.response.output
                    if output.get("diff"):
                        markdown += "#### Diff\n\n"
                        markdown += f"```diff\n{output['diff']}\n```\n"

                    if output.get("errors"):
                        markdown += "#### Errors\n\n"
                        markdown += f"{output['errors']}\n\n"

                    if output.get("message"):
                        markdown += "#### Message\n\n"
                        markdown += f"{output['message']}\n\n"

            if state.name == "ClarifyCodeChange":

                if action.request.scratch_pad:
                    markdown += f"*{action.request.scratch_pad}*"

                if action.response and action.response.output:
                    output = action.response.output
                    if output.get("start_line"):
                        markdown += f"\n* Start Line: {output['start_line']}\n"
                        markdown += f"\n* End Line: {output['end_line']}\n"

            if state.name == "Finished":
                markdown += f"*{action.request.thoughts}*\n"

            if state.name == "Rejected":
                markdown += f"*{action.request.thoughts}*\n"

    markdown += "## Alternative patches\n"
    for alternative in instance["resolved_by"]:
        markdown += f"### {alternative['name']}\n"
        markdown += f"```diff\n{alternative['patch']}\n```\n"

    return markdown
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
            state_name = step['state']
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
