import asyncio
import logging
import os
import pytest
import shutil
import time
from moatless.benchmark.evaluation_manager import EvaluationManager
from moatless.benchmark.evaluation_runner import EvaluationRunner
from moatless.benchmark.schema import EvaluationStatus, InstanceStatus
from moatless.context_data import moatless_dir

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set moatless dir to test evals dir
#moatless_dir.set("./test_evals")

@pytest.mark.asyncio
async def test_evaluation_manager():
    """
    Tests the full lifecycle of creating and running an evaluation through the EvaluationManager.
    """
    manager = EvaluationManager()
    logger.info("EvaluationManager created")
    
    dataset_name = "verified_mini"  #"easy"
    agent_id = "code_and_test"
    model_id = "gpt-4o-mini-2024-07-18"
    max_concurrent_instances = 10
    max_iterations = 15
    max_expansions = 1
    
    evaluation = manager.create_evaluation(
        dataset_name=dataset_name,
        model_id=model_id,
        agent_id=agent_id,
        max_iterations=max_iterations,
        max_expansions=max_expansions,
        max_concurrent_instances=max_concurrent_instances,
    )
    logger.info(f"Evaluation created: {evaluation.evaluation_name}")

    runner = EvaluationRunner(
            evaluation=evaluation,
            max_concurrent_instances=max_concurrent_instances,
    )
        
    try:
        await runner.run_evaluation()
        
        logger.info("Final state counts:")
        for status in InstanceStatus:
            count = sum(1 for i in evaluation.instances if i.status == status)
            logger.info(f"{status.value}: {count}")
        
        evaluated_count = sum(1 for i in evaluation.instances if i.status == InstanceStatus.EVALUATED)
        error_count = sum(1 for i in evaluation.instances if i.status == InstanceStatus.ERROR)
        
        assert evaluated_count > 0, "Some instances should complete successfully"
        assert error_count == 0, f"No instances should error out, but got {error_count} errors"
    
    except Exception as e:
        logger.exception("Error during evaluation")
        raise
