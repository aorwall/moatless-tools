import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from benchmark import swebench
from benchmark.swebench import get_spans, apply_patch
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.python import PythonParser
from moatless.codeblocks.utils import Colors
from moatless.settings import Settings
from moatless.types import Span
from moatless.utils.repo import setup_github_repo, checkout_commit
from tests.test_write_code_regressions import UpdateCodeCase, _run_regression_test


def run_regression(instance_data: dict, base_dir: str = "/tmp/repos"):
    print(
        f"{Colors.YELLOW}Running regression: {instance_data['instance_id']}{Colors.RESET}"
    )

    instance_id = instance_data["instance_id"]
    path = setup_github_repo(
        repo=instance_data["repo"],
        base_commit=instance_data["base_commit"],
        base_dir=base_dir,
    )

    parser = PythonParser(apply_gpt_tweaks=True)

    for file_path in instance_data["patch_diff_details"].keys():
        with open(os.path.join(path, file_path), "r") as f:
            original_code = f.read()

        original_block = parser.parse(original_code)

        apply_patch(path, instance_data["instance_id"], instance_data["patch"])
        with open(os.path.join(path, file_path), "r") as f:
            expected_updated_code = f.read()

        updated_block = parser.parse(expected_updated_code)

        checkout_commit(path, instance_data["base_commit"])

        diff_spans = []
        for diff in instance_data["patch_diff_details"][file_path]["diffs"]:
            diff_spans.append(
                Span(
                    start_line=diff["start_line_old"],
                    end_line=diff.get("end_line_old", diff["start_line_old"]),
                )
            )

        updated_block_cases = []

        updated_diff_spans = []
        new_lines = 0
        for diff_span in diff_spans:
            if diff_span.start_line < diff_span.end_line:
                updated_diff_spans.append(diff_span)
                continue

            start_line = diff_span.start_line + new_lines

            maybe_new_block = updated_block.find_first_by_start_line(start_line)
            if not maybe_new_block:
                updated_diff_spans.append(diff_span)
                continue

            parent_block = maybe_new_block.parent

            new_blocks = []

            if parent_block:
                if parent_block.sum_tokens() < Settings.coder.min_tokens_for_split_span:
                    trimmed_parent_block = parent_block.copy_with_trimmed_parents()
                    updated_block_cases.append(
                        UpdateCodeCase(
                            content=trimmed_parent_block.root().to_string(),
                            span=Span(
                                block_path=parent_block.full_path(),
                                start_line=parent_block.start_line,
                                end_line=parent_block.end_line,
                            ),
                            action="update",
                        )
                    )
                    continue

                for child_block in parent_block.children:
                    if (
                        child_block.start_line < start_line
                        or not child_block.is_indexed
                    ):
                        continue

                    existing_block = original_block.find_by_path(
                        child_block.full_path()
                    )
                    if not existing_block:
                        new_blocks.append(child_block)
                    else:
                        break

            if not new_blocks:
                updated_diff_spans.append(diff_span)
                continue

            for new_block in new_blocks:
                print(f"Adding new block: {new_block.path_string()}")
                new_block = new_block.copy_with_trimmed_parents()
                updated_block_cases.append(
                    UpdateCodeCase(
                        content=new_block.root().to_string(),
                        span=Span(
                            block_path=new_block.full_path()[:-1],
                            start_line=diff_span.start_line,
                            end_line=diff_span.end_line,
                        ),
                        action="add",
                    )
                )
                new_lines += new_block.end_line - new_block.start_line

        context_spans = get_spans(original_block, updated_diff_spans)
        for span in context_spans:
            print(f"Testing span: {span}")

            first_block_in_span = original_block.find_first_block_in_span(span)
            expected_block = find_closest_indexed_parent(first_block_in_span)

            print(
                f"Expected block: {expected_block.path_string()}, tokens: {expected_block.sum_tokens()}"
            )

            if expected_block.full_path():
                updated_child_block = updated_block.find_by_path(
                    expected_block.full_path()
                )
                if not updated_child_block:
                    raise AssertionError(
                        f"Block not found: {expected_block.full_path()}"
                    )
                updated_child_block = updated_child_block.copy_with_trimmed_parents()
            else:
                updated_child_block = updated_block.model_copy()

            too_large = (
                expected_block.sum_tokens() > Settings.coder.min_tokens_for_split_span
            )
            expected_span = None
            if too_large and expected_block.type in [
                CodeBlockType.CLASS,
                CodeBlockType.MODULE,
            ]:
                expected_span = Span(
                    block_path=expected_block.full_path(),
                    start_line=span.start_line,
                    end_line=span.end_line,
                    is_partial=True,
                )

                last_child_block = None
                for child_block in expected_block.children:
                    if child_block.start_line < span.start_line:
                        continue

                    last_child_block = child_block
                    if child_block.start_line > span.end_line:
                        break

                trimmed_child_blocks = []
                for child_block in updated_child_block.children:
                    if child_block.start_line < span.start_line:
                        continue

                    if child_block.is_indexed:
                        break

                    if (
                        child_block.end_line >= span.end_line
                        and child_block.to_string() == last_child_block.to_string()
                    ):  # TODO: Workaround until we got additions/subtractions in the git diff summary
                        break

                    trimmed_child_blocks.append(child_block)

                updated_child_block.children = trimmed_child_blocks
                update_code_case = UpdateCodeCase(
                    content=updated_child_block.root().to_string(),
                    span=expected_span,
                    action="update",
                )
            else:
                if too_large:
                    print(
                        f"{Colors.YELLOW} The expected {expected_block.type.lower()} {expected_block.path_string()} has {expected_block.sum_tokens()}. Might need to split it... {Colors.RESET}"
                    )

                update_code_case = UpdateCodeCase(
                    content=updated_child_block.root().to_string(),
                    span=Span(
                        block_path=expected_block.full_path(),
                        start_line=expected_block.start_line,
                        end_line=expected_block.end_line,
                    ),
                    action="update",
                )

            if update_code_case in updated_block_cases:
                print(f"Block already updated {expected_block.path_string()}")
                continue

            if expected_span:
                print(
                    f"Span to be updated: L{expected_span.start_line}-L{expected_span.end_line}"
                )
            else:
                print(f"Block to be updated: {expected_block.path_string()}")

            updated_block_cases.append(update_code_case)

        if not updated_block_cases:
            raise AssertionError("No updated block cases found")

        temp_dir = tempfile.TemporaryDirectory()
        with open(os.path.join(temp_dir.name, "original.py"), "w") as f:
            f.write(original_code)

        updated_block_cases = sorted(
            updated_block_cases, key=lambda x: x.span.start_line
        )

        try:
            _run_regression_test(
                updated_block_cases, expected_updated_code, temp_dir.name
            )
        except Exception as e:
            print(f"{Colors.RED}Regression failed: {instance_id}")
            print(
                f"{Colors.BLUE}Expected patch: {instance_data['patch']}{Colors.RESET}"
            )

            os.makedirs(
                f"../tests/data/python/regressions/{instance_id}", exist_ok=True
            )

            with open(
                f"../tests/data/python/regressions/{instance_id}/original.py", "w"
            ) as f:
                f.write(original_code)

            with open(
                f"../tests/data/python/regressions/{instance_id}/expected.py", "w"
            ) as f:
                f.write(expected_updated_code)

            with open(
                f"../tests/data/python/regressions/{instance_id}/updates.json", "w"
            ) as f:
                dicts = [case.dict() for case in updated_block_cases]
                json.dump(dicts, f, indent=2)

            raise e

    print(f"{Colors.GREEN}Regression passed: {instance_id}{Colors.RESET}")


def find_closest_indexed_parent(codeblock) -> Optional[CodeBlock]:
    indexed_parent = (
        find_closest_indexed_parent(codeblock.parent) if codeblock.parent else None
    )
    if (
        indexed_parent
        and indexed_parent.sum_tokens() < Settings.coder.min_tokens_for_split_span
    ):
        return indexed_parent

    if codeblock.is_indexed:
        return codeblock

    return indexed_parent


def run_instance(instance_id: str):
    instance_data = swebench.get_instance(
        instance_id,
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )
    run_regression(instance_data)


def test_single_instance():
    run_instance("django__django-11905")


def test_instances():
    instances = swebench.get_instances(
        dataset_name="princeton-nlp/SWE-bench_Lite",
        split="test",
        data_dir="../data",
    )

    start_at = None  # "django__django-13158"

    for instance_data in instances:
        if start_at and instance_data["instance_id"] != start_at:
            continue
        start_at = None
        try:
            run_regression(instance_data)
        except Exception as e:
            print(f"Failed: {instance_data['instance_id']}")
            raise e
