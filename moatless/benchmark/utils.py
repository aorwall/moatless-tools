import logging
import re
import time

from moatless.codeblocks.module import Module
from moatless.repository import FileRepository
from moatless.types import FileWithSpans

logger = logging.getLogger(__name__)


def find_relevant_spans(original_block: Module, updated_block: Module):
    """Find relevant spans in test content. Used for finding the "perfect" context in benchmark instances."""

    relevant_spans = set()

    for span in updated_block.spans_by_id.values():
        if span.span_id in relevant_spans:
            continue

        if original_block.has_span(span.span_id):
            updated_content = updated_block.to_prompt(
                span_ids=set(span.span_id), show_outcommented_code=False
            ).strip()
            original_content = original_block.to_prompt(
                span_ids=set(span.span_id), show_outcommented_code=False
            ).strip()
            if original_content != updated_content:
                relevant_spans.add(span.span_id)

            # TODO: Second prio after token count
            related_span_ids = original_block.find_related_span_ids(span.span_id)
            relevant_spans.update(related_span_ids)
        else:
            parent_block = updated_block.find_first_by_span_id(span.span_id).parent
            original_parent_block = original_block.find_by_path(
                parent_block.full_path()
            )
            span_ids = list(original_parent_block.belongs_to_span.span_id)

            related_span_ids = updated_block.find_related_span_ids(span.span_id)
            for related_span_id in related_span_ids:
                if original_block.has_span(related_span_id):
                    span_ids.append(related_span_id)

    return relevant_spans


def get_diff_lines(diff_input):
    if not diff_input:
        return []
    file_name_re = re.compile(r"diff --git a/(.+) b/.+")
    file_name_no_git_re = re.compile(r"--- a/(.+)")

    line_change_re = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

    changes = []

    current_file = None
    for line in diff_input.splitlines():
        file_match = file_name_re.match(line)
        if file_match:
            current_file = file_match.group(1)
            continue

        if not current_file:
            file_match = file_name_no_git_re.match(line)
            if file_match:
                current_file = file_match.group(1)

            continue

        line_change_match = line_change_re.match(line)
        if line_change_match:
            old_start, old_length, new_start, new_length = map(
                int, line_change_match.groups()
            )

            adjustment_start = max(1, min(3, old_start - 3))
            adjusted_start = old_start + adjustment_start

            relevant_diff_lines = max(0, old_length - 7)
            adjusted_end = adjusted_start + relevant_diff_lines

            changes.append((current_file, adjusted_start, adjusted_end))

    return changes


def compare_patches(expected_patch, actual_patch):
    expected_diffs = get_diff_lines(expected_patch)
    actual_diffs = get_diff_lines(actual_patch)

    expected_files = set()
    file_hits = set()
    line_hits = 0

    for patch_diff in expected_diffs:
        change_file, change_start, change_end = patch_diff

        for actual_diff in actual_diffs:
            actual_change_file, actual_change_start, actual_change_end = actual_diff
            expected_files.add(change_file)
            if change_file == actual_change_file:
                file_hits.add(change_file)
                if (
                    change_start >= actual_change_start
                    and change_end <= actual_change_end
                ):
                    line_hits += 1
                    continue

    return len(expected_files) - len(file_hits), len(expected_diffs) - line_hits


def create_file_spans_from_patch(repo_dir: str, patch: str) -> list[FileWithSpans]:
    repository = FileRepository(repo_dir)
    files_with_spans = []
    for file_path, span_ids in get_file_spans_from_patch(repository, patch).items():
        file_with_spans = FileWithSpans(
            file_path=file_path,
            span_ids=span_ids,
        )
        files_with_spans.append(file_with_spans)

    return files_with_spans


def get_file_spans_from_patch(
    repository: FileRepository, patch: str
) -> dict[str, list[str]]:
    expected_diff_lines = get_diff_lines(patch)
    expected_files_with_spans = {}

    for diff_line in expected_diff_lines:
        file = repository.get_file(diff_line[0])

        if file is None or file.module is None:
            continue

        if file.file_path not in expected_files_with_spans:
            expected_files_with_spans[file.file_path] = []

        spans = file.module.find_spans_by_line_numbers(diff_line[1], diff_line[2])
        for span in spans:
            if span.span_id not in expected_files_with_spans[file.file_path]:
                expected_files_with_spans[file.file_path].append(span.span_id)
    return expected_files_with_spans


def get_files_from_patch(patch: str) -> list[str]:
    diff_lines = get_diff_lines(patch)
    return [diff_line[0] for diff_line in diff_lines]


def file_spans_to_dict(files_with_spans: list[FileWithSpans]) -> dict[str, list[str]]:
    span_dict = {}
    if not files_with_spans:
        return span_dict

    for file_with_spans in files_with_spans:
        if file_with_spans.file_path not in span_dict:
            span_dict[file_with_spans.file_path] = []

        for span_id in file_with_spans.span_ids:
            if span_id not in span_dict[file_with_spans.file_path]:
                span_dict[file_with_spans.file_path].append(span_id)
    return span_dict


def get_missing_files(
    expected_files_with_spans: dict[str, list[str]],
    actual_files_with_spans: dict[str, list[str]],
) -> list[str]:
    misses = list(expected_files_with_spans.keys())
    for actual_file in actual_files_with_spans.keys():
        if actual_file in misses:
            misses.remove(actual_file)
    return misses


def get_missing_spans(
    expected_files_with_spans: dict[str, list[str]],
    actual_files_with_spans: dict[str, list[str]],
) -> dict[str, list[str]]:
    misses = {}
    for expected_file, span_ids in expected_files_with_spans.items():
        if expected_file not in actual_files_with_spans:
            misses[expected_file] = span_ids
            continue

        for span_id in span_ids:
            if span_id not in actual_files_with_spans[expected_file]:
                if expected_file not in misses:
                    misses[expected_file] = []
                misses[expected_file].append(span_id)

    return misses


def calculate_estimated_context_window(instance, results):
    patch_diffs = get_diff_lines(instance["patch"])
    expected_changes = []

    for patch_diff in patch_diffs:
        change_file, change_start, change_end = patch_diff
        expected_changes.append(
            {
                "file_path": change_file,
                "start_line": change_start,
                "end_line": change_end,
                "closest_match_context_window": None,
                "closest_match_lines": None,
                "position": None,
                "distance": None,
                "context_window": None,
            }
        )

    sum_tokens = 0

    for i, result in enumerate(results):
        sum_tokens += result.tokens
        for change in expected_changes:
            if result.file_path == change["file_path"]:
                if (
                    result.start_line - 1 <= change["start_line"]
                    and result.end_line + 1 >= change["end_line"]
                ):
                    change["distance"] = result.distance
                    change["context_window"] = sum_tokens
                    change["position"] = i

                    if all(
                        context["context_window"] is not None
                        for context in expected_changes
                    ):
                        return expected_changes, sum_tokens
                else:
                    closest_match_lines = change.get("closest_match_lines")
                    if (
                        not closest_match_lines
                        or abs(result.start_line - change["start_line"])
                        < abs(closest_match_lines[0] - change["start_line"])
                    ) or (
                        abs(result.end_line - change["end_line"])
                        == abs(closest_match_lines[0] - change["end_line"])
                    ):
                        change["closest_match_lines"] = (
                            result.start_line,
                            result.end_line,
                        )
                        change["closest_match_context_window"] = sum_tokens

    return expected_changes, sum_tokens


def get_total_cost(trace_id):
    try:
        import langfuse
    except ImportError:
        logger.info("Langfuse not installed, can't get total cost")
        return 0

    langfuse = langfuse.Langfuse()
    trace = langfuse.get_trace(trace_id)

    return trace.total_cost


def trace_metadata(instance_id: str, session_id: str, trace_name: str):
    date_time_str = time.strftime("%Y%m%d-%H%M%S")
    trace_id = f"coder_{instance_id}_{date_time_str}"
    return {
        "session_id": session_id,
        "name": trace_name,
        "trace": trace_name,
        "trace_id": trace_id,
        "tags": [instance_id],
    }
