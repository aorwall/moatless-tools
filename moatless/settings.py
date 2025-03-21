import asyncio
import os
import logging

from dotenv import load_dotenv

from moatless.eventbus.local_bus import LocalEventBus
from moatless.runner.asyncio_runner import AsyncioRunner
from moatless.storage.file_storage import FileStorage

load_dotenv()

logger = logging.getLogger(__name__)

if not os.environ.get("MOATLESS_DIR"):
    raise ValueError("MOATLESS_DIR environment variable is not set")

storage = None
event_bus = None
runner = None

model_manager = None
agent_manager = None
flow_manager = None
evaluation_manager = None


async def initialize_managers():
    """Initialize all manager instances asynchronously by calling their get_instance() methods.

    This function simply accesses the singleton instances, ensuring they're created.
    The actual singleton management is handled by the classes themselves.
    """
    from moatless.completion.manager import ModelConfigManager
    from moatless.agent.manager import AgentConfigManager
    from moatless.flow.manager import FlowManager
    from moatless.evaluation.manager import EvaluationManager

    global model_manager, agent_manager, flow_manager, evaluation_manager

    event_bus = await get_event_bus()
    runner = await get_runner()
    storage = await get_storage()

    try:
        logger.info("Initializing manager singletons...")

        model_manager = ModelConfigManager(storage=storage)
        agent_manager = AgentConfigManager(storage=storage)
        await model_manager.initialize()
        await agent_manager.initialize()

        if event_bus is None or runner is None:
            raise ValueError("Event bus or runner is not initialized")

        flow_manager = FlowManager(
            storage=storage,
            eventbus=event_bus,
            runner=runner,
            agent_manager=agent_manager,
            model_manager=model_manager,
        )
        await flow_manager.initialize()

        evaluation_manager = EvaluationManager(
            storage=storage,
            eventbus=event_bus,
            runner=runner,
            flow_manager=flow_manager,
        )
        await evaluation_manager.initialize()

        logger.info("All manager singletons accessed successfully")
    except Exception as e:
        logger.exception(f"Error initializing manager singletons: {e}")
        raise e


async def ensure_managers_initialized():
    """Ensure the manager singletons are initialized."""
    global model_manager, agent_manager, flow_manager, evaluation_manager
    if model_manager is None or agent_manager is None or flow_manager is None or evaluation_manager is None:
        await initialize_managers()


async def get_agent_manager():
    """Get the agent manager instance, ensuring it's initialized."""

    await ensure_managers_initialized()

    if agent_manager is None:
        raise RuntimeError("Agent manager not initialized")

    return agent_manager


async def get_model_manager():
    """Get the model manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if model_manager is None:
        raise RuntimeError("Model manager not initialized")

    return model_manager


async def get_flow_manager():
    """Get the flow manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if flow_manager is None:
        raise RuntimeError("Flow manager not initialized")

    return flow_manager


async def get_event_bus():
    """Get the event bus instance."""
    global event_bus

    if event_bus is None:
        if os.environ.get("REDIS_URL"):
            logger.info(f"Use RQ Runner and Redis Event Bus with redis url: {os.environ.get('REDIS_URL')}")
            try:
                from moatless.eventbus.redis_bus import RedisEventBus
                from moatless.runner.rq import RQRunner

                event_bus = RedisEventBus(redis_url=os.environ.get("REDIS_URL"), storage=storage)
            except Exception as e:
                logger.error(f"Failed to initialize event bus and runner: {e}")
                raise e
        else:
            logger.info("Use Local Runner and Local Event Bus")
            event_bus = LocalEventBus(storage=storage)

    await event_bus.initialize()

    return event_bus


async def get_runner():
    """Get the runner instance."""
    global runner

    if runner is None:
        runner_type = os.environ.get("MOATLESS_RUNNER")
        if runner_type == "kubernetes":
            from moatless.runner.kubernetes_runner import KubernetesRunner

            runner = KubernetesRunner()
        elif runner_type == "rq":
            from moatless.runner.rq import RQRunner

            runner = RQRunner(redis_url=os.environ.get("REDIS_URL"))
        else:
            logger.info("Use Local Runner and Local Event Bus")
            runner = AsyncioRunner()

    return runner


async def get_storage():
    """Get the storage instance."""
    global storage
    if storage is None:
        if os.environ.get("MOATLESS_STORAGE") == "s3":
            from moatless.storage.s3_storage import S3Storage

            storage = S3Storage()
        else:
            storage = FileStorage(base_dir=os.environ.get("MOATLESS_DIR"))

    return storage


async def get_evaluation_manager():
    """Get the evaluation manager instance, ensuring it's initialized."""
    await ensure_managers_initialized()

    if evaluation_manager is None:
        raise RuntimeError("Evaluation manager not initialized")

    return evaluation_manager
