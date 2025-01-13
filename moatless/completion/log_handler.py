import json
import logging
import os
from datetime import datetime

from litellm import CustomLogger

logger = logging.getLogger("LiteLLM-Logger")


class LogHandler(CustomLogger):
    def __init__(self, log_dir: str | None = None):
        super().__init__()
        self.log_dir = log_dir or f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)

    def _write_to_file(self, filename, data):
        from datetime import datetime
        import json

        now = datetime.now()
        timestamped_filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{filename}"

        # Calculate duration only if both start and end times exist
        if "duration" in data and data["duration"] is not None:
            if hasattr(data["duration"], "total_seconds"):
                data["duration"] = data["duration"].total_seconds()

        log_entry = {"timestamp": now.isoformat(), "data": data}

        # Check if instance_id exists in kwargs
        instance_id = None
        if (
            "kwargs" in data
            and "litellm_params" in data["kwargs"]
            and "metadata" in data["kwargs"]["litellm_params"]
        ):
            instance_id = data["kwargs"]["litellm_params"]["metadata"].get(
                "instance_id"
            )

        # Use instance_id directory if available, otherwise use default log_dir
        log_path = f"{self.log_dir}/{instance_id}" if instance_id else self.log_dir
        os.makedirs(log_path, exist_ok=True)

        try:
            with open(f"{log_path}/{timestamped_filename}", "a") as f:
                f.write(json.dumps(log_entry, default=str, indent=4) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        original_response = kwargs.get("original_response")
        response_content = None
        if original_response:
            try:
                # Replace escaped backslashes before parsing
                cleaned_response = original_response.replace("\\\\", "\\")
                original_response = json.loads(cleaned_response)

                if (
                    "choices" in original_response
                    and "message" in original_response["choices"][0]
                    and "content" in original_response["choices"][0]["message"]
                ):
                    try:
                        response_content = json.loads(
                            original_response["choices"][0]["message"]["content"]
                        )
                    except Exception as e:
                        response_content = original_response["choices"][0]["message"][
                            "content"
                        ]

            except Exception as e:
                original_response = str(original_response)

        data = {
            "response": response_obj,
            "original_response": original_response,
            "response_content": response_content,
            "kwargs": kwargs,
            "duration": (end_time - start_time) if (start_time and end_time) else None,
        }
        self._write_to_file("post_api_calls.json", data)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        data = {
            "response": response_obj,
            "kwargs": kwargs,
            "duration": (end_time - start_time) if (start_time and end_time) else None,
        }
        self._write_to_file("failure_events.json", data)
