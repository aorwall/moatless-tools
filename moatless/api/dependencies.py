"""Dependency functions for FastAPI routes."""

import asyncio
import logging
import os
import secrets
import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from moatless.agent.manager import AgentConfigManager
from moatless.completion.manager import ModelConfigManager
from moatless.evaluation.manager import EvaluationManager
from moatless.eventbus.base import BaseEventBus
from moatless.flow.manager import FlowManager
from moatless.runner.runner import BaseRunner
from moatless.storage.base import BaseStorage
import moatless.settings as settings

logger = logging.getLogger(__name__)

# Create a unique worker_id for this process
WORKER_ID = f"worker-{os.getpid()}-{uuid.uuid4().hex[:8]}"
logger.info(f"Initializing worker with ID: {WORKER_ID}")

# Module-level singleton instances
_storage = None
_event_bus = None
_runner = None
_model_manager = None
_agent_manager = None
_flow_manager = None
_evaluation_manager = None

# Lock for thread-safe initialization
_init_lock = asyncio.Lock()

# Track subscriptions to prevent duplicates
_event_subscriptions = set()


async def get_storage() -> BaseStorage:
    """Get the storage instance, initializing it if necessary."""
    global _storage

    # Fast path if already initialized
    if _storage is not None:
        return _storage

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _storage is None:
            _storage = await settings.get_storage()
            logger.info(f"Storage initialized for worker {WORKER_ID}")

    return _storage


async def get_event_bus() -> BaseEventBus:
    """Get the event bus instance, initializing it if necessary."""
    global _event_bus, _event_subscriptions

    # Fast path if already initialized
    if _event_bus is not None:
        return _event_bus

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _event_bus is None:
            _event_bus = await settings.get_event_bus()
            logger.info(f"Event bus initialized for worker {WORKER_ID}")

            # Import here to avoid circular imports
            from moatless.api.websocket import handle_event

            # Only subscribe if not already subscribed
            subscription_key = f"{handle_event.__qualname__}:{WORKER_ID}"
            if subscription_key not in _event_subscriptions:
                await _event_bus.subscribe(handle_event)
                _event_subscriptions.add(subscription_key)
                logger.info(f"Worker {WORKER_ID} subscribed to handle_event")

    return _event_bus


async def get_runner() -> BaseRunner:
    """Get the runner instance, initializing it if necessary."""
    global _runner

    # Fast path if already initialized
    if _runner is not None:
        return _runner

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _runner is None:
            _runner = await settings.get_runner()

    return _runner


async def get_model_manager(storage: BaseStorage = Depends(get_storage)) -> ModelConfigManager:
    """Get the model manager instance, initializing it if necessary."""
    global _model_manager

    # Fast path if already initialized
    if _model_manager is not None:
        return _model_manager

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _model_manager is None:
            _model_manager = ModelConfigManager(storage=storage)
            await _model_manager.initialize()
            logger.info(f"Model manager initialized for worker {WORKER_ID}")

    return _model_manager


async def get_agent_manager(storage: BaseStorage = Depends(get_storage)) -> AgentConfigManager:
    """Get the agent manager instance, initializing it if necessary."""
    global _agent_manager

    # Fast path if already initialized
    if _agent_manager is not None:
        return _agent_manager

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _agent_manager is None:
            _agent_manager = AgentConfigManager(storage=storage)
            await _agent_manager.initialize()
            logger.info(f"Agent manager initialized for worker {WORKER_ID}")

    return _agent_manager


async def get_flow_manager(
    storage: BaseStorage = Depends(get_storage),
    event_bus: BaseEventBus = Depends(get_event_bus),
    runner: BaseRunner = Depends(get_runner),
    agent_manager: AgentConfigManager = Depends(get_agent_manager),
    model_manager: ModelConfigManager = Depends(get_model_manager),
) -> FlowManager:
    """Get the flow manager instance, initializing it if necessary."""
    global _flow_manager

    # Fast path if already initialized
    if _flow_manager is not None:
        return _flow_manager

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _flow_manager is None:
            _flow_manager = FlowManager(
                storage=storage,
                eventbus=event_bus,
                runner=runner,
                agent_manager=agent_manager,
                model_manager=model_manager,
            )
            await _flow_manager.initialize()
            logger.info(f"Flow manager initialized for worker {WORKER_ID}")

    return _flow_manager


async def get_evaluation_manager(
    storage: BaseStorage = Depends(get_storage),
    event_bus: BaseEventBus = Depends(get_event_bus),
    runner: BaseRunner = Depends(get_runner),
    flow_manager: FlowManager = Depends(get_flow_manager),
) -> EvaluationManager:
    """Get the evaluation manager instance, initializing it if necessary."""
    global _evaluation_manager, _event_subscriptions

    # Fast path if already initialized
    if _evaluation_manager is not None:
        return _evaluation_manager

    # Slow path with lock to prevent race conditions
    async with _init_lock:
        # Check again in case another task initialized while we were waiting
        if _evaluation_manager is None:
            _evaluation_manager = EvaluationManager(
                storage=storage,
                eventbus=event_bus,
                runner=runner,
                flow_manager=flow_manager,
            )

            # Only initialize once to prevent duplicate event handling
            subscription_key = f"EvaluationManager._handle_event:{WORKER_ID}"
            if subscription_key not in _event_subscriptions:
                await _evaluation_manager.initialize()
                _event_subscriptions.add(subscription_key)
                logger.info(f"Evaluation manager initialized and subscribed to events for worker {WORKER_ID}")
            else:
                logger.info(f"Evaluation manager initialized for worker {WORKER_ID} (already subscribed)")

    return _evaluation_manager


# Function to clean up all resources
async def cleanup_resources():
    """Clean up all manager and service resources."""
    global \
        _event_bus, \
        _storage, \
        _runner, \
        _model_manager, \
        _agent_manager, \
        _flow_manager, \
        _evaluation_manager, \
        _event_subscriptions

    logger.info(f"Cleaning up resources for worker {WORKER_ID}...")

    # Unsubscribe from events
    if _event_bus:
        # Import here to avoid circular imports
        from moatless.api.websocket import handle_event

        subscription_key = f"{handle_event.__qualname__}:{WORKER_ID}"
        if subscription_key in _event_subscriptions:
            await _event_bus.unsubscribe(handle_event)
            _event_subscriptions.remove(subscription_key)
            logger.info(f"Worker {WORKER_ID} unsubscribed from handle_event")

        # Also close the event bus connection
        if hasattr(_event_bus, "close"):
            await _event_bus.close()
            logger.info(f"Event bus closed for worker {WORKER_ID}")

    # Reset all references
    _storage = None
    _event_bus = None
    _runner = None
    _model_manager = None
    _agent_manager = None
    _flow_manager = None
    _evaluation_manager = None
    _event_subscriptions.clear()

    logger.info(f"All resources cleaned up for worker {WORKER_ID}")
