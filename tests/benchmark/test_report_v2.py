import json
from pathlib import Path

import pytest

from moatless.benchmark.report_v2 import to_result
from moatless.trajectory import Trajectory


@pytest.fixture
def django_trajectory():
    file_path = Path("tests/trajectories/django__django_16379.json")
    return Trajectory.load(str(file_path))


@pytest.fixture
def dataset():
    with open("moatless/benchmark/swebench_lite_all_evaluations.json") as f:
        return json.load(f)

@pytest.fixture
def django_instance(dataset):
    for instance in dataset:
        if instance["instance_id"] == "django__django-16379":
            return instance

    return None


def test_to_result(django_trajectory, django_instance):
    result = to_result(django_instance, django_trajectory)

    assert result["instance_id"] == "django__django-16379"
    assert result["status"] == "edited"
    assert result["transitions"] == len(django_trajectory.transitions)
    assert result["edited"]
    assert result["identified"]
    assert result["found_in_search"]
    assert result["file_identified"]
