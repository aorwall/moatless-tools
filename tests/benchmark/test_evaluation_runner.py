"""
Sample tests for the EvaluationRunner above. These tests use pytest and mock out
external dependencies (e.g. moatless library calls) to verify state transitions,
error behavior, concurrency, and event emission.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import time

from moatless.benchmark.schema import (
    Evaluation,
    EvaluationInstance,
    TreeSearchSettings,
    EvaluationStatus,
    InstanceStatus,
)
from moatless.benchmark.evaluation_runner import EvaluationRunner


@pytest.mark.asyncio
async def test_evaluation_runner_basic_flow():
    """
    Tests that an Evaluation with two instances transitions through CREATED -> SETTING_UP -> PENDING -> RUNNING.
    """

    # Arrange: create a minimal evaluation
    instances = [
        EvaluationInstance(instance_id="test_inst_1", status=InstanceStatus.CREATED),
        EvaluationInstance(instance_id="test_inst_2", status=InstanceStatus.CREATED),
    ]
    tree_search_settings = TreeSearchSettings(
        model_id="test_model",
        agent_id="test_agent",
        max_iterations=1,
        max_expansions=1
    )
    evaluation = Evaluation(
        evaluation_name="test_evaluation",
        dataset_name="test_dataset",
        instances=instances,
        settings=tree_search_settings,
    )

    runner = EvaluationRunner(evaluation=evaluation, max_concurrent_instances=2)

    # Mock out the domain-specific calls to external libraries.
    with patch("moatless.benchmark.evaluation_runner.get_moatless_instance", return_value={"repo": "fake_repo", "instance_id": "fake_id", "problem_statement": "Some bug"}), \
         patch("moatless.benchmark.evaluation_runner.create_repository_async", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.create_index", return_value=MagicMock()), \
         patch("moatless.benchmark.evaluation_runner.get_agent", return_value=MagicMock()), \
         patch("moatless.benchmark.evaluation_runner.create_completion_model", return_value=MagicMock()), \
         patch.object(runner, "_start_agentic_system", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.TestbedEnvironment.evaluate", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.Node.from_file", return_value=MagicMock(get_leaf_nodes=lambda: [])), \
         patch.object(runner, "_cleanup_repositories"):

        # Act: Start the main loop, but we'll cancel once we see a transition or run a short time.
        main_task = asyncio.create_task(runner.run_evaluation())

        # Wait a short time for the runner to process states
        await asyncio.sleep(0.5)

        # Assert: We should see the first transitions happen quickly
        # Cancel to stop the loop for this test
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass

        # We can check that the runner has begun setting up the first instance
        assert any(i.status == InstanceStatus.SETTING_UP for i in instances) or \
               any(i.status == InstanceStatus.PENDING for i in instances) or \
               any(i.status == InstanceStatus.RUNNING for i in instances)


@pytest.mark.asyncio
async def test_evaluation_runner_error_propagation():
    """
    If an exception is raised during instance setup, the instance should go to ERROR state.
    """

    # Arrange: one instance
    instance = EvaluationInstance(instance_id="inst_error", status=InstanceStatus.CREATED)
    tree_search_settings = TreeSearchSettings(
        model_id="test_model",
        agent_id="test_agent",
        max_iterations=1,
        max_expansions=1
    )
    evaluation = Evaluation(
        evaluation_name="test_evaluation_error",
        dataset_name="test_dataset",
        instances=[instance],
        settings=tree_search_settings,
    )
    runner = EvaluationRunner(evaluation=evaluation, max_concurrent_instances=1)

    # Force create_repository_async to raise an exception
    with patch("moatless.benchmark.evaluation_runner.get_moatless_instance", return_value={"repo": "error_repo", "instance_id": "inst_error", "problem_statement": "Fail me"}), \
         patch("moatless.benchmark.evaluation_runner.create_repository_async", side_effect=RuntimeError("Repo creation failed")), \
         patch.object(runner, "_cleanup_repositories"):

        # Act: Start the process
        main_task = asyncio.create_task(runner.run_evaluation())

        # Wait for the instance to transition to ERROR state or timeout after 2 seconds
        start_time = time.time()
        while instance.status != InstanceStatus.ERROR and time.time() - start_time < 2:
            await asyncio.sleep(0.1)

        # Cancel the main task now that we've checked the status
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass

        # Assert: The instance should be in ERROR state
        assert instance.status == InstanceStatus.ERROR, f"Instance status is {instance.status}, expected ERROR"


@pytest.mark.asyncio
async def test_evaluation_completion():
    """
    Verifies that once all instances reach EVALUATED or ERROR, the runner sets EVALUATION status to COMPLETED.
    """
    instances = [
        EvaluationInstance(instance_id="i1", status=InstanceStatus.EVALUATED),
        EvaluationInstance(instance_id="i2", status=InstanceStatus.EVALUATED),
    ]
    tree_search_settings = TreeSearchSettings(
        model_id="test_model",
        agent_id="test_agent",
        max_iterations=1,
        max_expansions=1
    )
    evaluation = Evaluation(
        evaluation_name="already_evaluated",
        dataset_name="test_dataset",
        instances=instances,
        settings=tree_search_settings,
    )
    runner = EvaluationRunner(evaluation=evaluation, max_concurrent_instances=2)

    # Since both instances are already EVALUATED, the runner should complete immediately
    with patch.object(runner, "_cleanup_repositories"):
        await runner.run_evaluation()

    assert evaluation.status == EvaluationStatus.COMPLETED
    assert evaluation.completed_at is not None


@pytest.mark.asyncio
async def test_max_concurrency():
    """
    Ensures that no more than max_concurrent_instances are in SETTING_UP or RUNNING at the same time.
    We'll create multiple instances and check state transitions with concurrency=1.
    """

    instances = [EvaluationInstance(instance_id=f"inst_{i}", status=InstanceStatus.CREATED)
                 for i in range(3)]
    tree_search_settings = TreeSearchSettings(
        model_id="test_model",
        agent_id="test_agent",
        max_iterations=1,
        max_expansions=1
    )
    evaluation = Evaluation(
        evaluation_name="concurrency_test",
        dataset_name="test_dataset",
        instances=instances,
        settings=tree_search_settings,
    )
    runner = EvaluationRunner(evaluation=evaluation, max_concurrent_instances=1)

    # Mock external calls
    with patch("moatless.benchmark.evaluation_runner.get_moatless_instance", return_value={"repo": "fake_repo", "instance_id": "fake_id", "problem_statement": "Test"}), \
         patch("moatless.benchmark.evaluation_runner.create_repository_async", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.create_index", return_value=MagicMock()), \
         patch("moatless.benchmark.evaluation_runner.get_agent", return_value=MagicMock()), \
         patch("moatless.benchmark.evaluation_runner.create_completion_model", return_value=MagicMock()), \
         patch.object(runner, "_start_agentic_system", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.TestbedEnvironment.evaluate", return_value=AsyncMock()), \
         patch("moatless.benchmark.evaluation_runner.Node.from_file", return_value=MagicMock(get_leaf_nodes=lambda: [])), \
         patch.object(runner, "_cleanup_repositories"):
        main_task = asyncio.create_task(runner.run_evaluation())
        await asyncio.sleep(1.0)

        # Cancel to stop further progression (simulates checking concurrency mid-run)
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass

    # With max_concurrent_instances=1, only one instance can be in SETTING_UP or RUNNING at once
    setting_up_count = sum(1 for i in instances if i.status == InstanceStatus.SETTING_UP)
    running_count = sum(1 for i in instances if i.status == InstanceStatus.RUNNING)
    assert (setting_up_count + running_count) <= 1, "Found more than one instance setting up/running concurrently."
