import json
import logging
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel

from moatless.context_data import current_node_id

import moatless.settings as settings

logger = logging.getLogger("LiteLLM-Logger")

IGNORE_KWARGS = ["original_response", "messages", "api_key", "additional_args", "tools"]


class LogHandler(CustomLogger):
    def __init__(self):
        super().__init__()
        self._storage = settings.storage

    async def _get_log_key(self, filename):
        now = datetime.now()
        node_id = current_node_id.get()

        if node_id:
            log_key = f"completions/node_{node_id}_{filename}"
            counter = 0
            while await self._storage.exists(log_key):
                counter += 1
                log_key = f"completions/node_{node_id}_retry_{counter}_{filename}"
        else:
            timestamped_filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{filename}"
            log_key = f"completions/{timestamped_filename}"

        return log_key

    async def _write_to_file_async(self, filename, data):
        now = datetime.now()
        log_entry = {"timestamp": now.isoformat(), "data": data}
        log_key = await self._get_log_key(filename)

        try:
            await self._storage.write_to_trajectory(key=log_key, data=log_entry)
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        original_response = self.parse_response(kwargs.get("original_response"))
        kwargs = {k: self._handle_kwargs_item(v) for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        await self._write_to_file_async("post_api_calls", data)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        original_response = kwargs.get("original_response")
        kwargs = {k: self._handle_kwargs_item(v) for k, v in kwargs.items() if k not in IGNORE_KWARGS}
        data = {
            "response": original_response,
            **kwargs,
        }
        await self._write_to_file_async("failure_events", data)

    def parse_response(self, original_response):
        if original_response:
            try:
                cleaned_response = original_response.replace("\\\\", "\\")
                original_response = json.loads(cleaned_response)
            except Exception:
                original_response = str(original_response)
        return original_response

    def _handle_kwargs_item(self, item):
        if isinstance(item, BaseModel):
            return item.model_dump()
        elif isinstance(item, dict):
            return {k: self._handle_kwargs_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._handle_kwargs_item(i) for i in item]
        return item
