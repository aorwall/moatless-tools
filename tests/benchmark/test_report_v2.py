import json
from pathlib import Path
import pandas as pd

import pytest

from moatless.benchmark.report_v2 import EditStats, PlanStats, SearchStats, StateStats, to_result, BenchmarkResult, to_dataframe
from moatless.benchmark.utils import get_moatless_instance
from moatless.trajectory import Trajectory


@pytest.fixture
def django_trajectory():
    file_path = Path("tests/trajectories/django__django_16379.json")
    return Trajectory.load(str(file_path))

@pytest.fixture
def scikit_trajectory():
    file_path = Path("tests/trajectories/scikit-learn__scikit-learn-13779/trajectory.json")
    return Trajectory.load(str(file_path))


@pytest.fixture
def dataset():
    with open("moatless/benchmark/swebench_lite_all_evaluations.json") as f:
        return json.load(f)


@pytest.fixture
def django_instance(dataset):
    return get_moatless_instance("django__django-16379", split="lite")


@pytest.fixture
def scikit_instance(dataset):
    return get_moatless_instance("scikit-learn__scikit-learn-13779", split="verified")


def test_to_result(django_trajectory, django_instance):
    result = to_result(django_instance, django_trajectory)

    assert isinstance(result, BenchmarkResult)
    assert result.instance_id == "django__django-16379"
    assert result.status == "edited"
    assert result.transitions == len(django_trajectory.transitions)
    assert result.edit.edited
    assert result.identify.status in ["expected_spans", "alternative_spans"]
    assert result.search.status in ["expected_spans", "alternative_spans"]
    assert result.identify.found_files > 0

    # New assertions to test the updated structure
    assert isinstance(result.expected_spans_details, dict)
    assert isinstance(result.search.found_spans_details, dict)
    assert isinstance(result.identify.found_spans_details, dict)
    assert isinstance(result.plan.found_spans_details, dict)
    assert isinstance(result.edit.found_spans_details, dict)

    assert result.expected_spans > 0
    assert result.expected_files > 0
    assert result.search.found_spans > 0
    assert result.search.found_files > 0
    assert result.identify.found_spans > 0
    assert result.identify.found_files > 0

    assert len(result.search.found_spans_details) > 0
    assert len(result.identify.found_spans_details) > 0

    assert result.search.found_spans == 1
    assert result.search.found_files == 1
    assert result.identify.found_spans == 1
    assert result.identify.found_files == 1

    assert result.expected_spans == 1
    assert result.expected_files == 1


def test_scikit_not_edited(scikit_trajectory, scikit_instance):
    result = to_result(scikit_instance, scikit_trajectory)

    print(json.dumps(result.model_dump(), indent=2))

    assert result.edit.status == "expected_files"



def test_to_result_resolved_case(django_trajectory, django_instance):
    # Simulate a resolved case
    report = {"resolved_ids": [django_instance["instance_id"]]}
    result = to_result(django_instance, django_trajectory, report)

    assert result.status == "resolved"
    assert result.resolved

@pytest.fixture
def sample_results():
    return [
        BenchmarkResult(
            instance_id="test1",
            duration=10.5,
            total_cost=0.05,
            resolved_by=1,
            status="resolved",
            transitions=5,
            all_transitions=7,
            expected_spans=3,
            expected_files=2,
            alternative_solutions=1,
            resolved=True,
            error="",
            search=SearchStats(
                status="expected_spans",
                iterations=2,
                rejected=0,
                cost=0.02,
                found_spans=3,
                found_files=2,
                result_spans=4,
                result_files=2,
                found_spans_details={"file1.py": ["span1", "span2"], "file2.py": ["span3"]},
                p_query=1,
                p_file=1,
                p_code=0,
                p_class=0,
                p_function=1
            ),
            identify=StateStats(
                status="expected_spans",
                iterations=1,
                rejected=0,
                cost=0.01,
                found_spans=3,
                found_files=2,
                result_spans=3,
                result_files=2,
                found_spans_details={"file1.py": ["span1", "span2"], "file2.py": ["span3"]}
            ),
            plan=PlanStats(
                status="",
                iterations=1,
                rejected=0,
                cost=0.01,
                found_spans=0,
                found_files=0,
                result_spans=0,
                result_files=0,
                found_spans_details={},
                review=True
            ),
            edit=EditStats(
                status="",
                iterations=1,
                rejected=0,
                cost=0.01,
                found_spans=0,
                found_files=0,
                result_spans=0,
                result_files=0,
                found_spans_details={},
                retries=0,
                edited=True,
                has_diff=True,
                lint=False,
                lints=""
            )
        )
    ]
