import os
from datetime import datetime

import pytest
from dotenv import load_dotenv

from moatless.benchmark.evaluation import Evaluation
from moatless.edit import PlanToCode, EditCode
from moatless.find.search import SearchCode
from moatless.find.identify import IdentifyCode
from moatless.find.decide import DecideRelevance
from moatless.transitions import search_and_code_transitions

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")
index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
repo_dir = os.getenv("REPO_DIR", "/tmp/repo")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",  # "azure/gpt-4o",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

state_params = {
    SearchCode: {
        "provide_initial_context": True,
        "max_search_results": 75,
        "initial_context_tokens": 6000,
        "initial_search_results": 100,
        "initial_context_spans_per_file": 5,
    },
    IdentifyCode: {"expand_context": True},
    DecideRelevance: {
        "finish_after_relevant_count": 1,
    },
    PlanToCode: {
        "max_tokens_in_edit_prompt": 750,
        "expand_context_with_related_spans": False,
        "finish_on_review": True,
    },
    EditCode: {
        "chain_of_thought": False,
        "show_file_context": False,
        "max_prompt_file_tokens": 8000,
    },
}

search_and_code = search_and_code_transitions(
    global_params=global_params, state_params=state_params
)

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.llm_integration
def test_run_single_evaluation_mcts():
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f"{moatless_dir}/eval_test"
    evaluation_name = f"{datestr}_mcts"

    evaluation = Evaluation(
        transitions=search_and_code,
        evaluations_dir=dir,
        evaluation_name=evaluation_name,
        index_store_dir=index_store_dir,
        repo_base_dir=repo_dir,
        max_file_context_tokens=16000,
        num_workers=1,
        detailed_report=True,
        report_modes=["search_and_identify"],
    )

    result = evaluation.run_single_instance("django__django-16379")

    assert result["instance_id"] == "django__django-16379"
    assert result["status"] == "edited"
    assert result["edited"]
    assert result["identified"]
    assert result["found_in_search"]
    assert result["file_identified"]

import os
import pandas as pd
import pytest
from moatless.benchmark.evaluation import Evaluation
from moatless.benchmark.report_v2 import BenchmarkResult, SearchStats, StateStats, PlanStats, EditStats
from moatless.transitions import search_and_code_transitions

@pytest.fixture
def sample_results():
    return [
        BenchmarkResult(
            instance_id="test_1",
            duration=10.5,
            total_cost=0.05,
            resolved_by=2,
            transitions=5,
            all_transitions=7,
            expected_spans=3,
            expected_files=2,
            expected_spans_details={"file1.py": ["span1", "span2"], "file2.py": ["span3"]},
            alternative_solutions=1,
            resolved=True,
            status="edited",
            search=SearchStats(found_spans=2, found_files=1, found_spans_details={"file1.py": ["span1", "span2"]}),
            identify=StateStats(found_spans=2, found_files=1, found_spans_details={"file1.py": ["span1", "span2"]}),
            plan=PlanStats(found_spans=1, found_files=1, found_spans_details={"file1.py": ["span1"]}),
            edit=EditStats(found_spans=1, found_files=1, found_spans_details={"file1.py": ["span1"]}, edited=True),
        ),
        BenchmarkResult(
            instance_id="test_2",
            duration=15.0,
            total_cost=0.07,
            resolved_by=1,
            transitions=6,
            all_transitions=8,
            expected_spans=2,
            expected_files=1,
            expected_spans_details={"file3.py": ["span4", "span5"]},
            alternative_solutions=0,
            resolved=False,
            status="identified",
            search=SearchStats(found_spans=1, found_files=1, found_spans_details={"file3.py": ["span4"]}),
            identify=StateStats(found_spans=2, found_files=1, found_spans_details={"file3.py": ["span4", "span5"]}),
            plan=PlanStats(),
            edit=EditStats(),
        )
    ]

def test_to_csv_report(tmp_path, sample_results):
    # Create a temporary directory for the test
    eval_dir = tmp_path / "test_evaluation"
    eval_dir.mkdir()

    # Initialize Evaluation object
    evaluation = Evaluation(
        evaluations_dir=str(tmp_path),
        evaluation_name="test_evaluation",
        transitions=search_and_code_transitions(),
        report_mode="search_and_identify"
    )

    # Call _to_csv_report
    evaluation._to_csv_report(sample_results)

    # Check if the CSV file was created
    csv_path = eval_dir / "result.csv"
    assert csv_path.exists()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check the number of rows
    assert len(df) == 2

    # Check if all expected columns are present
    expected_columns = [
        "instance_id", "duration", "total_cost", "resolved_by", "status",
        "transitions", "all_transitions", "expected_spans", "expected_files",
        "expected_spans_details", "resolved", "error",
        "iterations", "rejected", "cost", "found_spans", "found_files",
        "result_spans", "result_files", "found_spans_details"
    ]
    for col in expected_columns:
        assert col in df.columns

    # Check some specific values
    assert df.loc[0, "instance_id"] == "test_1"
    assert df.loc[0, "status"] == "edited"
    assert df.loc[1, "instance_id"] == "test_2"
    assert df.loc[1, "status"] == "identified"

    # Check that the spans details are properly serialized
    assert isinstance(df.loc[0, "expected_spans_details"], str)
    assert isinstance(df.loc[0, "found_spans_details"], str)

    # Verify that the report mode is applied correctly
    assert "p_query" in df.columns  # This is specific to SearchStats
    assert "review" not in df.columns  # This is specific to PlanStats and should not be included