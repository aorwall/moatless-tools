import json
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd

from moatless.repository import FileRepository
from moatless.benchmark.swebench import (
    found_in_expected_spans,
    found_in_alternative_spans,
    setup_swebench_repo,
)
from moatless.benchmark.utils import (
    has_identified_spans,
    has_identified_files, count_identified_files, count_identified_spans,
)
from moatless.file_context import FileContext, RankedFileSpan
from moatless.trajectory import Trajectory
from moatless.schema import VerificationError
from moatless.state import AgenticState, Content
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StateStats(BaseModel):
    status: str = ""
    iterations: int = 0
    rejected: int = 0
    cost: float = 0
    found_spans: int = 0
    found_files: int = 0
    result_spans: int = 0
    result_files: int = 0
    found_spans_details: Dict[str, List[str]] = {}


class SearchStats(StateStats):
    p_query: int = 0
    p_file: int = 0
    p_code: int = 0
    p_class: int = 0
    p_function: int = 0


class PlanStats(StateStats):
    review: bool = False


class EditStats(StateStats):
    retries: int = 0
    edited: bool = False
    has_diff: bool = False
    lint: bool = False
    lints: str = ""


class BenchmarkResult(BaseModel):
    instance_id: str
    duration: float = 0
    total_cost: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    resolved_by: int = 0
    transitions: int = 0
    all_transitions: int = 0
    expected_spans: int = 0
    expected_files: int = 0
    expected_spans_details: Dict[str, List[str]] = {}
    alternative_solutions: int = 0
    resolved: bool = False
    error: str = ""
    status: str = ""
    search: SearchStats = SearchStats()
    identify: StateStats = StateStats()
    plan: PlanStats = PlanStats()
    edit: EditStats = EditStats()
    decide: StateStats = StateStats()


def to_result(
    instance: Dict, trajectory: Trajectory, report: Optional[Dict] = None
) -> BenchmarkResult:
    info = trajectory._info

    if (
        report
        and "resolved_ids" in report
        and instance["instance_id"] in report["resolved_ids"]
    ):
        result_status = "resolved"
    else:
        result_status = info.get("status")

    resolved = result_status == "resolved"

    selected_transition_ids = []
    current_state = trajectory.get_current_state()
    while current_state:
        selected_transition_ids.append(current_state.id)
        current_state = current_state.previous_state

    logger.info(f"Selected transitions: {selected_transition_ids}")

    try:
        expected_spans = instance.get("expected_spans", {})
        expected_files = list(expected_spans.keys())
        expected_spans_details = expected_spans

        alternative_solutions = []
        for resolved_by in instance.get("resolved_by", []):
            if (
                "alternative_spans" in resolved_by
                and resolved_by["alternative_spans"] not in alternative_solutions
            ):
                alternative_solutions.append(resolved_by["alternative_spans"])

        result = BenchmarkResult(
            instance_id=instance["instance_id"],
            duration=info.get("duration", 0),
            total_cost=info.get("total_cost", 0),
            prompt_tokens=info.get("prompt_tokens", 0),
            completion_tokens=info.get("completion_tokens", 0),
            resolved_by=len(instance.get("resolved_by", [])),
            transitions=len(selected_transition_ids),
            all_transitions=len(trajectory.transitions),
            expected_spans=sum(len(spans) for spans in expected_spans.values()),
            expected_files=len(expected_files),
            expected_spans_details=expected_spans_details,
            alternative_solutions=len(alternative_solutions),
            resolved=resolved,
            status="",  # Initialize status
            search=SearchStats(),
            identify=StateStats(),
            plan=PlanStats(),
            edit=EditStats(),
            decide=StateStats(),
        )

        lint_codes = set()
        search_results_spans: Dict[str, List[str]] = {}
        identified_spans: Dict[str, List[str]] = {}
        planned_spans: Dict[str, List[str]] = {}
        edited_spans: Dict[str, List[str]] = {}

        if expected_spans:
            for transition in trajectory.transitions:
                if (
                    selected_transition_ids
                    and transition.id not in selected_transition_ids
                ):
                    continue

                state: AgenticState = transition.state
                state_name = state.name

                # Update iterations and cost for the specific state
                if state_name in ["search", "identify", "decide", "plan", "edit"]:
                    current_state_stats = getattr(result, state_name)
                    current_state_stats.iterations += 1

                    if current_state.response.trigger == "reject":
                        current_state_stats.rejected += 1

                    if state._actions:
                        for action in state._actions:
                            if action.completion and action.completion.usage:
                                current_state_stats.cost += (
                                    action.completion.usage.completion_cost
                                )

                    # Update the state stats in the result object
                    setattr(result, state_name, current_state_stats)

                if state_name == "SearchCode":
                    if state.action_request:
                        for search_request in state.action_request.search_requests:
                            if search_request.query:
                                result.search.p_query += 1
                            if search_request.file_pattern:
                                result.search.p_file += 1
                            if search_request.code_snippet:
                                result.search.p_code += 1
                            if search_request.class_names:
                                result.search.p_class += 1
                            if search_request.function_names:
                                result.search.p_function += 1

                    if state.outcome and "ranked_spans" in state.outcome:
                        for ranked_span in state.outcome["ranked_spans"]:
                            if isinstance(ranked_span, RankedFileSpan):
                                ranked_span = ranked_span.model_dump()

                            result.search.result_spans += 1
                            if ranked_span["file_path"] not in search_results_spans:
                                search_results_spans[ranked_span["file_path"]] = []
                                result.search.result_files += 1
                            search_results_spans[ranked_span["file_path"]].append(
                                ranked_span["span_id"]
                            )

                    result.search.found_spans = sum(len(spans) for spans in search_results_spans.values())
                    result.search.found_files = len(search_results_spans)
                    result.search.found_spans_details = search_results_spans
                    set_found_status(
                        expected_spans,
                        alternative_solutions,
                        search_results_spans,
                        result.search,
                    )

                if state_name == "IdentifyCode" and state.action_request:
                    for span in state.action_request.identified_spans:
                        result.identify.result_spans += 1

                        if span.file_path not in identified_spans:
                            identified_spans[span.file_path] = []
                            result.identify.result_files += 1

                        for span_id in span.span_ids:
                            identified_spans[span.file_path].append(span_id)

                    set_found_status(
                        expected_spans,
                        alternative_solutions,
                        identified_spans,
                        result.identify,
                    )

                if state_name == "PlanToCode" and state.action_request:
                    if state.action_request.action == "review":
                        result.plan.review = True

                    if state.action_request.file_path:
                        file_path = state.action_request.file_path
                        if file_path not in planned_spans:
                            planned_spans[file_path] = []
                        planned_spans[file_path].append(state.action_request.span_id)

                    set_found_status(
                        expected_spans,
                        alternative_solutions,
                        planned_spans,
                        result.plan,
                    )

                if state_name == "EditCode" and state.action_request:
                    result.edit.retries = len(state._actions) - 1

                    edited = state.response and state.response.trigger == "finish"

                    if edited and hasattr(state, "file_path"):
                        file_path = state.file_path
                        if file_path not in edited_spans:
                            edited_spans[file_path] = []
                        edited_spans[file_path].append(state.span_id)

                    if not result.edit.edited and (
                        found_in_expected_spans(instance, edited_spans)
                        or found_in_alternative_spans(instance, edited_spans)
                    ):
                        result.edit.edited = True

                    if state.response and state.response.output:
                        output = state.response.output
                        if edited:
                            result.edit.has_diff = True

                        try:
                            if output.get("verification_errors"):
                                result.edit.lint = True
                                for lint in output.get("verification_errors", []):
                                    if isinstance(lint, VerificationError):
                                        lint_codes.add(lint.code)
                                    else:
                                        lint_codes.add(lint["code"])
                                result.edit.lints = ",".join(lint_codes)
                        except Exception as e:
                            logger.exception(f"Failed to parse lint code from {output}")

                    set_found_status(
                        expected_spans, alternative_solutions, edited_spans, result.edit
                    )

        if "error" in info:
            result.error = info["error"].split("\n")[0]
        else:
            result.error = ""

        if result.resolved:
            result.status = "resolved"
        elif result.edit.status in ["expected_spans", "alternative_spans"]:
            result.status = "edited"
        elif result.plan.status in ["expected_spans", "alternative_spans"]:
            result.status = "planned"
        elif result.identify.status in ["expected_spans", "alternative_spans"]:
            result.status = "identified"
        elif result.search.status in ["expected_spans", "alternative_spans"]:
            result.status = "searched"
        elif "error" in info:
            result.status = "error"
        else:
            result.status = "failed"

    except Exception as e:
        raise e

    logger.info(f"Result: {result}")

    return result


def set_found_status(
    expected_spans, alternative_solutions, identified_spans, result_stats: StateStats
):
    result_stats.result_spans = sum(len(spans) for spans in identified_spans.values())
    result_stats.result_spans = len(identified_spans)
    result_stats.found_files = count_identified_files(expected_spans, identified_spans)
    result_stats.found_spans = count_identified_spans(expected_spans, identified_spans)
    result_stats.found_spans_details = identified_spans

    expected_files = list(expected_spans.keys())
    if result_stats.found_spans == sum(len(spans) for spans in expected_spans.values()):
        result_stats.status = "expected_spans"
    elif has_identified_spans(alternative_solutions, identified_spans):
        result_stats.status = "alternative_spans"
    elif result_stats.found_files == len(expected_files):
        result_stats.status = "expected_files"
    elif has_identified_files(alternative_solutions, identified_spans):
        result_stats.status = "alternative_files"
    else:
        result_stats.status = "missing_spans"


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


def to_dataframe(report_mode: str, results: list[BenchmarkResult]) -> pd.DataFrame:
    state_keys = ["search", "identify", "decide", "plan", "edit"]
    rename_columns = False
    if report_mode == "code":
        state_keys = ["plan", "edit"]
    elif report_mode == "search_and_identify":
        state_keys = ["search", "identify"]
    elif report_mode in state_keys:
        state_keys = [report_mode]
        rename_columns = True

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        general_keys = ["instance_id", "duration", "total_cost", "resolved_by", "status",
                        "transitions", "all_transitions", "alternative_solutions", "resolved",
                        "expected_spans", "expected_files", "error"]

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if new_key.split(sep)[0] in state_keys or new_key in general_keys:
                if new_key in state_keys and isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))

            if k.endswith('_spans_details'):
                items.append((new_key, json.dumps(v)))
        return dict(items)

    flattened_results = [flatten_dict(result.model_dump()) for result in results]

    df = pd.DataFrame(flattened_results)

    if rename_columns:
        df.columns = [
            col.replace(f"{report_mode}_", "")
            if col.startswith(f"{report_mode}_")
            else col
            for col in df.columns
        ]

    # Reorder columns
    column_order = [
        "instance_id", "duration", "total_cost", "promt_tokens", "completion_tokens", "resolved_by", "status", "resolved",
        "transitions", "all_transitions", "expected_spans", "expected_files", "alternative_solutions",
        "expected_spans_details", "error"
    ]

    state_columns = ["status", "iterations", "rejected", "cost", "found_spans", "found_files",
                     "result_spans", "result_files", "found_spans_details"]

    for state in state_keys:
        column_order.extend([f"{state}_{col}" for col in state_columns])

    # Add any remaining columns
    remaining_columns = [col for col in df.columns if col not in column_order]
    column_order.extend(remaining_columns)

    # Reorder the dataframe columns
    df = df.reindex(columns=[col for col in column_order if col in df.columns])
    return df

def read_results_from_json(file_path: str) -> List[BenchmarkResult]:
    with open(file_path, "r") as f:
        data = json.load(f)

    results = [BenchmarkResult.validate(item) for item in data]
    return results
