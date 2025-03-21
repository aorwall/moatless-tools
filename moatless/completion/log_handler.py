import json
import logging
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel

from moatless.context_data import current_node_id, current_action_step
from moatless.storage.base import BaseStorage


logger = logging.getLogger("LiteLLM-Logger")

IGNORE_KWARGS = ["response", "api_key", "async_complete_streaming_response"]


class LogHandler(CustomLogger):
    def __init__(self, storage: BaseStorage):
        super().__init__()
        self._storage = storage

    async def _get_log_key(self, filename: str | None = None):
        now = datetime.now()
        node_id = current_node_id.get()
        action_step = current_action_step.get()

        trajectory_key = self._storage.get_trajectory_key()

        if not filename:
            filename = "completion"

        log_key = f"{trajectory_key}/completions/"

        if node_id:
            if action_step is not None:
                log_key += f"action_{node_id}_{action_step}"
            else:
                log_key += f"node_{node_id}"

            if node_id:
                counter = 1
                retry_log_key = f"{log_key}_call_{counter}"

                while await self._storage.exists(retry_log_key):
                    counter += 1
                    retry_log_key = f"{log_key}_call_{counter}"

                log_key = retry_log_key
        else:
            log_key += f"{now.strftime('%Y%m%d_%H%M%S')}_{filename}"

        return log_key

    async def _write_to_file_async(self, data: dict):
        log_key = await self._get_log_key()

        try:
            await self._storage.write(key=log_key, data=data)
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("response"):
            original_response = self.parse_response(kwargs.get("response"))
        elif kwargs.get("original_response"):
            original_response = self.parse_response(kwargs.get("original_response"))
        else:
            logger.debug(f"No response found in kwargs: {kwargs}")
            original_response = None

        if "additional_args" in kwargs and kwargs.get("additional_args").get("complete_input_dict"):
            original_input = kwargs.get("additional_args").get("complete_input_dict")
        else:
            original_input = {
                "messages": kwargs.get("input", []),
                **kwargs.get("optional_params", {}),
            }

        data = {
            "start_time": start_time,
            "end_time": end_time,
            "original_response_obj": original_response,
            "original_response": self._handle_kwargs_item(kwargs.get("async_complete_streaming_response")),
            "original_input": self._handle_kwargs_item(original_input),
            "litellm_response": self._handle_kwargs_item(response_obj),
        }
        await self._write_to_file_async(data)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self.async_log_success_event(kwargs, response_obj, start_time, end_time)

    def parse_response(self, original_response):
        if not original_response:
            return None
        if isinstance(original_response, str):
            try:
                cleaned_response = original_response.replace("\\\\", "\\")
                original_response = json.loads(cleaned_response)
            except Exception:
                original_response = str(original_response)
        else:
            return self._handle_kwargs_item(original_response)

        return original_response

    def _handle_kwargs_item(self, item):
        if isinstance(item, BaseModel):
            return item.model_dump()
        elif isinstance(item, dict):
            return {k: self._handle_kwargs_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._handle_kwargs_item(i) for i in item]
        return item
