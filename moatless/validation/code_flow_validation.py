#!/usr/bin/env python3
"""Base script for running integration tests and generating result summaries."""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import litellm
from dotenv import load_dotenv

from moatless.runner import agentic_runner
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.model import Usage
from moatless.loop import AgenticLoop
from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import create_repository
from moatless.index import CodeIndex
from moatless.config.model_config import create_completion_model
from moatless.config.agent_config import create_agent
from moatless.completion.log_handler import LogHandler
from moatless.events import BaseEvent, event_bus

class CodeFlowValidation:
    def __init__(self):
        self.base_dir = os.getenv("MOATLESS_DIR", ".moatless/runs")
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def get_run_dir(self, run_id: str) -> str:
        """Get the directory path for a validation run."""
        return os.path.join(self.base_dir, run_id)

    def setup_run_directory(self, run_dir: str) -> dict:
        """Create directory structure for the validation run."""
        dirs = {
            'root': run_dir,
            'logs': os.path.join(run_dir, 'logs'),
            'prompt_logs': os.path.join(run_dir, 'prompt_logs'),
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        prompt_log_callback = LogHandler(log_dir=dirs['prompt_logs'])
        litellm.callbacks = [prompt_log_callback]
        
        return dirs

    def start_code_loop(self, 
                       run_id: str,
                       agent_id: str,
                       model_id: str,
                       instance_id: str,
                       max_iterations: int = 15) -> str:
        """Run validation using CodeFlowValidation."""
        run_dir = self.get_run_dir(run_id)
        self.setup_run_directory(run_dir)
        trajectory_file = os.path.join(run_dir, 'trajectory.json')
        
        instance = get_moatless_instance(instance_id)
        repository = create_repository(instance)

        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
        code_index = CodeIndex.from_index_name(
            instance_id,
            index_store_dir=index_store_dir,
            file_repo=repository,
        )

        runtime = None
        if os.getenv("TESTBED_BASE_URL") and os.getenv("TESTBED_API_KEY"):
            from moatless.runtime.testbed import TestbedEnvironment
            runtime = TestbedEnvironment(
                repository=repository,
                instance_id=instance_id,
            )

        completion_model = create_completion_model(model_id)
        completion_model.metadata = {"instance_id": instance_id}
        
        agent = create_agent(
            config_id=agent_id,
            completion_model=completion_model,
            repository=repository,
            code_index=code_index,
            runtime=runtime,
        )

        loop = AgenticLoop.create(
            message=f"<task>\n{instance['problem_statement']}\n</task>",
            run_id=run_id,
            agent=agent,
            max_iterations=max_iterations,
            persist_path=trajectory_file,
            persist_dir=run_dir,
            metadata={
                "instance_id": instance_id,
                "model_id": model_id,
                "agent_id": agent_id
            }
        )

        agentic_runner.start(loop)
        return run_id
