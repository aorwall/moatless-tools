import asyncio
import json
import logging
from datetime import datetime

from litellm.integrations.custom_logger import CustomLogger
from moatless.context_data import current_node_id, current_action_step, current_phase
from moatless.storage.base import BaseStorage
from pydantic import BaseModel

logger = logging.getLogger()

IGNORE_KWARGS = ["response", "api_key", "async_complete_streaming_response"]


class LogHandler(CustomLogger):
    def __init__(self, storage: BaseStorage):
        super().__init__()
        self._storage = storage

    async def _get_log_path(self, filename: str | None = None):
        now = datetime.now()
        node_id = current_node_id.get()
        action_step = current_action_step.get()
        phase = current_phase.get()

        trajectory_key = self._storage.get_trajectory_path()

        if not filename:
            filename = "completion"

        log_path = f"{trajectory_key}/completions/"

        if node_id:
            log_path += f"node_{node_id}"
            if action_step is not None:
                log_path += f"/action_{action_step}"
            elif phase:
                log_path += f"/{phase}"
            else:
                log_path += f"/agent"

            if node_id:
                counter = 1
                retry_log_path = f"{log_path}/{counter}.json"

                while await self._storage.exists(retry_log_path):
                    counter += 1
                    retry_log_path = f"{log_path}/{counter}.json"

                log_path = retry_log_path
        else:
            log_path += f"{now.strftime('%Y%m%d_%H%M%S')}_{filename}"

        return log_path

    async def _write_to_file_async(self, data: dict):
        log_path = await self._get_log_path()

        try:
            await self._storage.write(path=log_path, data=data)
        except Exception as e:
            logger.exception(f"Failed to write log to {log_path}. Data: {data}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
    
        response_key = "async_complete_streaming_response"
        if response_key not in kwargs:
            response_key = "original_response"
            
        original_response = kwargs.get(response_key)

        if not original_response:
            logger.warning(f"No {response_key} found in kwargs with keys: {kwargs.keys()}")
            
            if response_obj:
                original_response = response_obj
            else:
                original_response = f"Couldn't find LLM response. No {response_key} found in LiteLLM completion kwargs with keys: {kwargs.keys()}"
            
        original_response = self.parse_response(original_response)
         
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
            "original_response": original_response,
            "original_input": self._handle_kwargs_item(original_input),
            "litellm_response": self._handle_kwargs_item(response_obj),
        }

        if "traceback_exception" in kwargs:
            data["traceback_exception"] = self._handle_kwargs_item(kwargs["traceback_exception"])

        if "exception" in kwargs:
            data["exception"] = self._handle_kwargs_item(kwargs["exception"])

        await self._write_to_file_async(data)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        logger.warning(f"Processing failure event for node {current_node_id.get()}")
        # Handle coroutine in response_obj for failure events
        if hasattr(response_obj, "__class__") and response_obj.__class__.__name__ == "coroutine":
            logger.warning("Got coroutine in failure response_obj, converting to string")
            response_obj = str(response_obj)
        await self.async_log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        logger.info(f"Processing stream event for node {current_node_id.get()}")
        if hasattr(response_obj, "__class__") and response_obj.__class__.__name__ == "coroutine":
            logger.warning("Got coroutine in stream response_obj, converting to string")
            response_obj = str(response_obj)
        await self.async_log_success_event(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        logger.warning(f"Processing failure event for node {current_node_id.get()}")
        if hasattr(response_obj, "__class__") and response_obj.__class__.__name__ == "coroutine":
            logger.warning("Got coroutine in failure response_obj, converting to string")
            response_obj = str(response_obj)
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.async_log_success_event(kwargs, response_obj, start_time, end_time))
        except Exception as e:
            logger.exception(f"Failed to log stream event for node {current_node_id.get()}")

    def parse_response(self, original_response):
        if not original_response:
            return None
        
        if hasattr(original_response, "__class__") and original_response.__class__.__name__ == "coroutine":
            logger.warning(f"Got coroutine in original_response, converting to string")
            original_response = str(original_response)
        
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
        if isinstance(item, str):
            return item
        elif isinstance(item, BaseModel):
            return item.model_dump()
        elif isinstance(item, dict):
            return {k: self._handle_kwargs_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [self._handle_kwargs_item(i) for i in item]
        elif isinstance(item, (set, tuple)):
            return [self._handle_kwargs_item(i) for i in item]
        elif hasattr(item, "__class__") and item.__class__.__name__ == "coroutine":
            return str(item)  # Convert coroutine objects to string representation
        elif isinstance(item, Exception):
            return {
                "error_type": item.__class__.__name__,
                "error_message": str(item),
                "traceback": getattr(item, "__traceback__", None) and str(item.__traceback__),
            }
        elif isinstance(item, datetime):
            return item.isoformat()
        elif isinstance(item, bytes) or isinstance(item, bytearray):
            return item.decode("utf-8", errors="replace")
        # Catch-all for other non-serializable objects
        try:
            json.dumps(item)
            return item
        except (TypeError, OverflowError):
            return str(item)
