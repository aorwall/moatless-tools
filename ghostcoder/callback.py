import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class LogCallbackHandler(BaseCallbackHandler):

    def __init__(self, log_dir: str = 'prompts') -> None:
        """Initialize callback handler."""
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def log_to_file(self, file_name: str, text: str) -> None:
        with open(self.log_dir + "/" + file_name, 'w') as f:
            f.write(text)
            f.write('\n')

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_to_file(filename_time + "_on_llm_start.log", "method: on_llm_start [{}]\nserialized: {}\nprompt:\n{}\n".format(current_time, json.dumps(serialized), prompts[0].replace("\\n", "\n")))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_to_file(filename_time + "_on_llm_end.log", "method: on_llm_end [{}]\nllm_output: {}\nprompt:\n{}\n---"
                         .format(current_time, json.dumps(response.llm_output), response.generations[0][0].text))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_to_file(filename_time + "_on_llm_error.log", "method: on_llm_error  [{}]\nerror: {}".format(current_time, str(error)))

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""
        #self.log_to_file({'method': 'on_chain_error', 'error': str(error)})

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing."""

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Do nothing."""

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Do nothing."""
