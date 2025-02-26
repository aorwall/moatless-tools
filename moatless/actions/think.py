from typing import List
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext
from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation

# https://github.com/sierra-research/tau-bench/blob/14bf0ef52e595922d597a38f32d3e8c0dce3a8f8/tau_bench/envs/retail/tools/think.py

class ThinkArgs(ActionArguments):
    """Use the tool to think about something. It will not obtain new information or change the database,
    but just append the thought to the log. Use it when complex reasoning is needed.
    """

    thought: str = Field(..., description="Your chain of thought reasoning.")

    model_config = ConfigDict(title="Think")

    @property
    def log_name(self):
        return f"Think({self.thought})"

    def to_prompt(self):
        return f"Thinking about:\n{self.thought}"

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return []

class Think(Action):
    args_schema = ThinkArgs
    
    async def execute(self, args: ThinkArgs, file_context: FileContext):
        return Observation(message="")
