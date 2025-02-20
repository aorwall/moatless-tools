import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from moatless.evaluation.schema import (
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
)
from moatless.evaluation.runner import EvaluationRunner
from testbeds.schema import EvalTestResult, EvaluationResult, TestsStatus
from testbeds.swebench.constants import ResolvedStatus

@pytest.fixture
def evaluation():
    """Create a test evaluation with two instances"""
    instances = [
        EvaluationInstance(instance_id="django__django-11099"),
        EvaluationInstance(instance_id="django__django-13658"),
    ]
    return Evaluation(
        evaluation_name="test_eval",
        dataset_name="easy",
        flow_id="test_flow",
        model_id="test_model",
        instances=instances,
    )

@pytest.fixture
def runner(evaluation, tmp_path):
    """Create an EvaluationRunner with test directories"""
    return EvaluationRunner(
        evaluation=evaluation,
        repo_base_dir=str(tmp_path / "repos"),
        evaluations_dir=str(tmp_path / "evals"),
        num_concurrent_instances=2
    )

@pytest.mark.asyncio
async def test_basic_state_transitions(runner):
    """Test that instances progress through expected states"""
    
    # Mock all external dependencies
    with patch("moatless.benchmark.report.create_trajectory_stats") as mock_create_stats, \
         patch("moatless.benchmark.swebench.create_index") as mock_create_index, \
         patch("moatless.evaluation.runner.create_flow") as mock_create_flow, \
         patch("moatless.evaluation.runner.TestbedEnvironment") as mock_testbed, \
         patch("moatless.evaluation.runner.Node.from_file") as mock_node, \
         patch("moatless.evaluation.runner.event_bus.publish") as mock_publish:

        # Create mock stats that won't trigger the encoding error
        mock_stats = MagicMock()
        mock_stats.status = "completed"
        mock_stats.resolved = True
        mock_create_stats.return_value = mock_stats

        mock_create_index.return_value = MagicMock()

        # Setup flow mocks
        flow_mock = AsyncMock()
        flow_mock.run = AsyncMock(return_value={"success": True})
        mock_create_flow.return_value = flow_mock

        # Setup testbed mocks
        testbed_mock = AsyncMock()
        testbed_mock.evaluate = AsyncMock(return_value=EvaluationResult(
            resolved=True,
            instance_id="django__django-11099",
            tests_status=TestsStatus(
                status=ResolvedStatus.FULL,
                fail_to_pass=EvalTestResult(success=["test1", "test2"]),
                pass_to_pass=EvalTestResult(success=["test3", "test4"])
            )
        ))
        mock_testbed.return_value = testbed_mock

        # Setup node mocks for evaluation
        node_mock = MagicMock()
        node_mock.get_leaf_nodes.return_value = []
        mock_node.return_value = node_mock

        # Start runner
        task = asyncio.create_task(runner.run_evaluation())
        
        # Wait for evaluation to complete or timeout
        try:
            await asyncio.wait_for(task, timeout=60.0)  # Increased timeout
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Verify state transitions
        assert runner.evaluation.status == EvaluationStatus.COMPLETED
        
        # Count instances in each state
        states = {i.status: i.status for i in runner.evaluation.instances}
        assert len(states) == 1, f"Expected all instances in same state, got: {states}"
        assert list(states.keys())[0] == InstanceStatus.EVALUATED

        # Verify repository was set up
        assert runner.state_manager._active_repos == {}

        # Verify flow was executed
        flow_mock.run.assert_called()
        
        # Verify evaluation happened
        testbed_mock.evaluate.assert_called()

        # Verify events were emitted
        mock_publish.assert_any_call(pytest.approx({
            "evaluation_name": "test_eval",
            "event_type": "evaluation_started"
        }, rel=1e-3))
        mock_publish.assert_any_call(pytest.approx({
            "evaluation_name": "test_eval", 
            "event_type": "evaluation_completed"
        }, rel=1e-3))

@pytest.mark.asyncio
async def test_repository_setup_tracking(runner):
    """Test that repository setup is properly tracked"""
    
    with patch("moatless.evaluation.runner.get_moatless_instance") as mock_get_instance, \
         patch("moatless.evaluation.runner.repository_exists") as mock_repo_exists, \
         patch.object(runner, "create_repository_async") as mock_create_repo, \
         patch("moatless.evaluation.runner.event_bus.publish"):

        mock_get_instance.return_value = {"repo": "test_repo", "instance_id": "test_1"}
        mock_repo_exists.return_value = False
        mock_create_repo.side_effect = RuntimeError("Setup failed")

        # Run evaluation
        with pytest.raises(RuntimeError):
            await runner.run_evaluation()

        # Verify instance went to error state
        instance = runner.evaluation.get_instance("test_1")
        assert instance.status == InstanceStatus.ERROR
        assert "Setup failed" in instance.error

        # Verify repo setup tracking was cleaned up
        assert not runner.state_manager._active_repos

@pytest.mark.asyncio
async def test_concurrent_instance_limits(runner):
    """Test that concurrent instance limits are respected"""
    
    # Set concurrency to 1
    runner.num_concurrent_instances = 1
    
    with patch("moatless.evaluation.runner.get_moatless_instance") as mock_get_instance, \
         patch("moatless.evaluation.runner.repository_exists") as mock_repo_exists, \
         patch.object(runner, "create_repository_async") as mock_create_repo, \
         patch("moatless.evaluation.runner.event_bus.publish"):

        mock_get_instance.return_value = {"repo": "test_repo", "instance_id": "test_id"}
        mock_repo_exists.return_value = True
        
        # Make repository creation take some time
        async def slow_create(*args, **kwargs):
            await asyncio.sleep(0.5)
            return MagicMock()
        mock_create_repo.side_effect = slow_create

        # Start evaluation
        task = asyncio.create_task(runner.run_evaluation())
        await asyncio.sleep(0.1)  # Let it start processing

        # Check that only one instance is running
        running = sum(1 for i in runner.evaluation.instances 
                     if i.status in (InstanceStatus.RUNNING, InstanceStatus.SETTING_UP))
        assert running <= 1

        # Cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_evaluation_persistence(runner, tmp_path):
    """Test that evaluation state is properly persisted"""
    
    with patch("moatless.evaluation.runner.get_moatless_instance"), \
         patch.object(runner, "create_repository_async"), \
         patch("moatless.evaluation.runner.event_bus.publish"):

        # Start evaluation
        task = asyncio.create_task(runner.run_evaluation())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify evaluation file was created
        eval_file = Path(tmp_path) / "evals" / runner.evaluation.evaluation_name / "evaluation.json"
        assert eval_file.exists()

@pytest.mark.asyncio
async def test_error_handling_and_cleanup(runner):
    """Test error handling and cleanup behavior"""
    
    with patch("moatless.evaluation.runner.get_moatless_instance") as mock_get_instance, \
         patch.object(runner, "create_repository_async") as mock_create_repo, \
         patch("moatless.evaluation.runner.event_bus.publish") as mock_publish, \
         patch("shutil.rmtree") as mock_rmtree:

        mock_get_instance.return_value = {"repo": "test_repo", "instance_id": "test_1"}
        mock_create_repo.side_effect = Exception("Test error")

        # Run evaluation
        await runner.run_evaluation()

        # Verify error state
        assert runner.evaluation.status == EvaluationStatus.ERROR
        assert "Test error" in runner.evaluation.error

        # Verify error event was emitted
        mock_publish.assert_any_call(pytest.approx({
            "evaluation_name": "test_eval",
            "event_type": "evaluation_error",
            "data": {"error": "Test error"}
        }, rel=1e-3))

        # Verify cleanup was called
        mock_rmtree.assert_called()

@pytest.mark.asyncio
async def test_prevent_duplicate_transitions(runner):
    """Test that we don't get stuck in duplicate state transitions"""
    
    with patch("moatless.evaluation.runner.get_moatless_instance") as mock_get_instance, \
         patch("moatless.benchmark.swebench.utils.repository_exists") as mock_repo_exists, \
         patch("moatless.evaluation.runner.event_bus.publish"):

        mock_get_instance.return_value = {"repo": "test_repo", "instance_id": "django__django-11099"}
        mock_repo_exists.return_value = True

        instance = runner.evaluation.instances[0]
        
        # First transition should work
        await runner.state_manager.set_status(instance, InstanceStatus.SETTING_UP)
        assert instance.status == InstanceStatus.SETTING_UP
        
        # Second attempt at same transition should be ignored
        await runner.state_manager.set_status(instance, InstanceStatus.SETTING_UP)
        assert instance.status == InstanceStatus.SETTING_UP  # Status unchanged
        
        # Move to PENDING
        await runner.state_manager.set_status(instance, InstanceStatus.PENDING)
        assert instance.status == InstanceStatus.PENDING
        
        # Trying SETTING_UP again should be ignored since we've done this transition before
        await runner.state_manager.set_status(instance, InstanceStatus.SETTING_UP)
        assert instance.status == InstanceStatus.PENDING  # Should not change
        
        # Different transitions should still work
        await runner.state_manager.set_status(instance, InstanceStatus.RUNNING)
        assert instance.status == InstanceStatus.RUNNING

@pytest.mark.asyncio
async def test_error_state_transitions(runner):
    """Test that transitions to ERROR state are always allowed"""
    
    with patch("moatless.evaluation.runner.get_moatless_instance") as mock_get_instance, \
         patch("moatless.benchmark.swebench.utils.repository_exists") as mock_repo_exists, \
         patch("moatless.evaluation.runner.event_bus.publish"):

        mock_get_instance.return_value = {"repo": "test_repo", "instance_id": "django__django-11099"}
        mock_repo_exists.return_value = True

        instance = runner.evaluation.instances[0]
        
        # First transition
        await runner.state_manager.set_status(instance, InstanceStatus.SETTING_UP)
        assert instance.status == InstanceStatus.SETTING_UP
        
        # Can transition to ERROR from any state
        await runner.state_manager.set_status(instance, InstanceStatus.ERROR, "Test error")
        assert instance.status == InstanceStatus.ERROR
        assert instance.error == "Test error"
        
        # Can transition to ERROR multiple times with different errors
        await runner.state_manager.set_status(instance, InstanceStatus.ERROR, "Another error")
        assert instance.status == InstanceStatus.ERROR
        assert instance.error == "Another error" 