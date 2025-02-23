import json
import logging
import os
from datetime import datetime
from pathlib import Path
import aiofiles

from moatless.context_data import current_node_id
from moatless.context_data import current_trajectory_id

from litellm import CustomLogger

from moatless.utils.moatless import get_moatless_trajectories_dir, get_moatless_trajectory_dir

logger = logging.getLogger("LiteLLM-Logger")

IGNORE_KWARGS = ["original_response", "messages", "api_key", "additional_args", "tools"]


class LogHandler(CustomLogger):
    def __init__(self, log_dir: str | None = None, trajectory_dir: str | None = None):
        super().__init__()
        
        self.trajectory_dir = trajectory_dir
        
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = None

        logger.info(f"Log handler initialized with log_dir: {self.log_dir} and trajectory_dir: {self.trajectory_dir}")

    def _get_log_path(self, filename):
        now = datetime.now()
        node_id = current_node_id.get()
        trajectory_id = current_trajectory_id.get()

        if self.log_dir:
            log_dir = f"{self.log_dir}/{trajectory_id}/completions" if trajectory_id else self.log_dir
        else:
            if self.trajectory_dir:
                log_dir = self.trajectory_dir
            elif trajectory_id:
                log_dir = get_moatless_trajectory_dir(trajectory_id) / "completions"
            else:
                log_dir = get_moatless_trajectories_dir() / "completions"

        if not os.path.exists(log_dir):
            logger.debug(f"Creating log directory: {log_dir}")
            os.makedirs(log_dir, exist_ok=True)

        if node_id:
            log_path = f"{log_dir}/node_{node_id}_{filename}"
            counter = 0
            while os.path.exists(log_path):
                counter += 1
                log_path = f"{log_dir}/node_{node_id}_retry_{counter}_{filename}"
        else:
            timestamped_filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{filename}"
            log_path = f"{log_dir}/{timestamped_filename}"

        return log_path

    def _write_to_file(self, filename, data):
        now = datetime.now()
        log_entry = {"timestamp": now.isoformat(), "data": data}
        log_path = self._get_log_path(filename)

        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry, default=str, indent=4) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    async def _write_to_file_async(self, filename, data):
        now = datetime.now()
        log_entry = {"timestamp": now.isoformat(), "data": data}
        log_path = self._get_log_path(filename)

        try:
            async with aiofiles.open(log_path, "a") as f:
                await f.write(json.dumps(log_entry, default=str, indent=4) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        original_response = self.parse_response(kwargs.get("original_response"))
        kwargs = {k: v for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        self._write_to_file("post_api_calls.json", data)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        original_response = kwargs.get("original_response")
        kwargs = {k: v for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        self._write_to_file("failure_events.json", data)

    async def async_log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        original_response = self.parse_response(kwargs.get("original_response"))
        kwargs = {k: v for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        await self._write_to_file_async("post_api_calls.json", data)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        original_response = kwargs.get("original_response")
        kwargs = {k: v for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        await self._write_to_file_async("failure_events.json", data)

    def parse_response(self, original_response):
        if original_response:
            try:
                cleaned_response = original_response.replace("\\\\", "\\")
                original_response = json.loads(cleaned_response)
            except Exception as e:
                original_response = str(original_response)
        return original_response
