import json
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd

from moatless.edit.plan import Review, RequestCodeChange
from moatless.index.code_index import is_test
from moatless.benchmark.utils import (
    has_identified_spans,
    has_identified_files, count_identified_files, count_identified_spans, get_missing_files,
)
from moatless.file_context import FileContext, RankedFileSpan
from moatless.trajectory import Trajectory
from moatless.schema import VerificationIssue
from moatless.state import AgenticState, Content, Rejected
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


class CodingStats(StateStats):
    review: bool = False
    retries: int = 0
    edited: bool = False

    rejected: int = 0
    largest_span: Optional[int] = None
    smallest_span: Optional[int] = None
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

    expected_test_files: List[str] = []
    found_test_files: List[str] = []
    missing_test_files: int = 0

    max_verification_issues: int = 0
    final_verification_issues: int = 0

    test_count: int = 0
    fail_to_pass_count: int = 0
    pass_to_pass_count: int = 0

    alternative_solutions: int = 0
    resolved: bool = False
    error: str = ""
    status: str = ""

    possible_issues: List[str] = []

    search: SearchStats = SearchStats()
    identify: StateStats = StateStats()
    coding: CodingStats = CodingStats()
    decide: StateStats = StateStats()


def to_result(
    instance: Dict, trajectory: Trajectory, report: Optional[Dict] = None
) -> BenchmarkResult:
    info = trajectory._info

    selected_transition_ids = []
    current_state = trajectory.get_current_state()
    while current_state:
        selected_transition_ids.append(current_state.id)
        current_state = current_state.previous_state

    logger.debug(f"Selected transitions: {selected_transition_ids}")

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

        if info.get("resolved", False):
            status = "resolved"
        elif info.get("error"):
            status = "error"
        elif isinstance(trajectory.get_current_state(), Rejected):
            status = "rejected"
        elif info.get("status") == "finished":
            status = "failed"
        else:
            status = "running"

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
            status=status,
            search=SearchStats(),
            identify=StateStats(),
            coding=CodingStats(),
            decide=StateStats(),
        )

        search_results_spans: Dict[str, List[str]] = {}
        identified_spans: Dict[str, List[str]] = {}
        pinned_spans: Dict[str, List[str]] = {}
        test_files: List[str] = []

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
                    if not state.action_request.identified_spans:
                        logger.warning(f"No action request found in IdentifyCode state: {state}")
                    else:
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
                    if isinstance(state.action_request.action, Review):
                        result.coding.review = True

                    if state.verification_issues and len(state.verification_issues) > result.max_verification_issues:
                        result.max_verification_issues = len(state.verification_issues)

                    result.final_verification_issues = len(state.verification_issues) if state.verification_issues else 0

                if state_name == "EditCode" and state.action_request:
                    if len(state._actions) > 1:
                        result.coding.retries += 1

                    if not result.coding.largest_span or state.end_line - state.start_line > result.coding.largest_span:
                        result.coding.largest_span = state.end_line - state.start_line

                    if not result.coding.smallest_span or state.end_line - state.start_line < result.coding.smallest_span:
                        result.coding.smallest_span = state.end_line - state.start_line

                    if transition.state.response.trigger == "reject":
                        result.coding.rejected += 1

                    if state.response and state.response.output:
                        output = state.response.output
                        if output.get("diff"):
                            result.coding.has_diff = True

                if state_name in ["Finished", "Rejected"]:
                    for file in transition.snapshot["file_context"]["files"]:
                        if is_test(file["file_path"]):
                            test_files.append(file["file_path"])
                            continue

                        pinned_spans = {}
                        for span in file["spans"]:
                            if not span.get("pinned", False):
                                continue

                            if file["file_path"] not in pinned_spans:
                                pinned_spans[file["file_path"]] = []

                            pinned_spans[file["file_path"]].append(span["span_id"])

        missing_tests = get_missing_files(
            instance["test_file_spans"], test_files
        )
        result.missing_test_files = len(missing_tests)
        result.expected_test_files = list(instance["test_file_spans"].keys())

        result.found_test_files = test_files

        if "evaluation_result" in info:
            test_status = info["evaluation_result"]["tests_status"]
            result.fail_to_pass_count = len(test_status["fail_to_pass"]["failure"])
            result.pass_to_pass_count = len(test_status["pass_to_pass"]["failure"])
            result.test_count = (len(test_status["fail_to_pass"]["failure"])
                                 + len(test_status["pass_to_pass"]["failure"])
                                 + len(test_status["fail_to_pass"]["success"])
                                 + len(test_status["pass_to_pass"]["success"]))

        set_found_status(
            expected_spans,
            alternative_solutions,
            pinned_spans,
            result.coding,
        )

        if "error" in info:
            result.error = info["error"].split("\n")[0]
        else:
            result.error = ""

    except Exception as e:
        raise e

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


def to_dataframe(results: list[BenchmarkResult], report_mode: str | None = None) -> pd.DataFrame:
    state_keys = ["search", "identify", "decide", "coding"]
    rename_columns = False
    if report_mode == "code":
        state_keys = ["coding"]
    elif report_mode == "search_and_identify":
        state_keys = ["search", "identify"]
    elif report_mode in state_keys:
        state_keys = [report_mode]
        rename_columns = True

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        general_keys = ["instance_id", "duration", "total_cost", "prompt_tokens", "completion_tokens", "resolved_by", "status",
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

    summary_cols = ["instance_id", "duration", "total_cost", "status", "transitions", "expected_spans", "expected_files", "search_status", "search_iterations", "identify_status", "identify_iterations", "decide_status", "decide_iterations", "plan_status", "plan_iterations", "edit_status", "edit_iterations"]

    flattened_results = [flatten_dict(result.model_dump()) for result in results]

    df = pd.DataFrame(flattened_results)

    if rename_columns:
        df.columns = [
            col.replace(f"{report_mode}_", "")
            if col.startswith(f"{report_mode}_")
            else col
            for col in df.columns
        ]

    if report_mode is None:
        df = df[summary_cols]

    # Reorder columns
    column_order = [
        "instance_id", "duration", "total_cost", "prompt_tokens", "completion_tokens", "resolved_by", "status", "resolved",
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
