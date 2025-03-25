"""Dependency functions for FastAPI routes."""

import logging
import os
import secrets

import moatless.settings as settings
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from moatless.agent.manager import AgentConfigManager
from moatless.completion.manager import ModelConfigManager
from moatless.evaluation.manager import EvaluationManager
from moatless.eventbus.base import BaseEventBus
from moatless.flow.manager import FlowManager
from moatless.runner.runner import BaseRunner
from moatless.storage.base import BaseStorage

logger = logging.getLogger(__name__)


async def get_agent_manager() -> AgentConfigManager:
    """Get the agent manager instance, ensuring it's initialized."""

    await settings.ensure_managers_initialized()

    if settings.agent_manager is None:
        raise RuntimeError("Agent manager not initialized")

    return settings.agent_manager


async def get_model_manager() -> ModelConfigManager:
    """Get the model manager instance, ensuring it's initialized."""
    await settings.ensure_managers_initialized()

    if settings.model_manager is None:
        raise RuntimeError("Model manager not initialized")

    return settings.model_manager


async def get_flow_manager() -> FlowManager:
    """Get the flow manager instance, ensuring it's initialized."""
    await settings.ensure_managers_initialized()

    if settings.flow_manager is None:
        raise RuntimeError("Flow manager not initialized")

    return settings.flow_manager


async def get_evaluation_manager() -> EvaluationManager:
    """Get the evaluation manager instance, ensuring it's initialized."""
    await settings.ensure_managers_initialized()

    if settings.evaluation_manager is None:
        raise RuntimeError("Evaluation manager not initialized")

    return settings.evaluation_manager
