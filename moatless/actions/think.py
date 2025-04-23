from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments, Observation, RewardScaleEntry
from moatless.completion.schema import FewShotExample
from moatless.file_context import FileContext


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
    def get_few_shot_examples(cls) -> list[FewShotExample]:
        return []


class Think(Action):
    args_schema = ThinkArgs

    async def execute(self, args: ThinkArgs, file_context: FileContext):
        return Observation.create(message="The thought was logged")

    @classmethod
    def get_evaluation_criteria(cls, trajectory_length: int | None = None) -> list[str]:
        """
        Get evaluation criteria specific to the Think action.

        Args:
            trajectory_length: The current trajectory length

        Returns:
            A list of evaluation criteria strings
        """
        return [
            "Depth of Reasoning: Evaluate the thoroughness and depth of the reasoning process, looking for consideration of multiple aspects, potential challenges, and edge cases.",
            "Structure and Clarity: Assess whether the reasoning is well-structured, clear, and follows a logical progression of thoughts.",
            "Problem Understanding: Determine if the reasoning demonstrates a comprehensive understanding of the problem being addressed.",
            "Appropriate Timing: Evaluate whether the Think action is used at appropriate decision points in the workflow (beginning of tasks, complex decisions, strategy formulation).",
            "Relevance: Assess whether the reasoning focuses on relevant aspects of the current problem rather than tangential topics.",
            "Action Planning: Check if the reasoning leads to a clear plan of action with specific next steps.",
            "Alternative Consideration: Evaluate whether the reasoning considers multiple approaches or solutions before selecting a path forward.",
            "Technical Accuracy: Determine if the technical content of the reasoning is accurate and demonstrates domain expertise.",
        ]

    @classmethod
    def get_reward_scale(cls, trajectory_length: int | None = None) -> list[RewardScaleEntry]:
        """
        Get reward scale specific to the Think action.

        Args:
            trajectory_length: The current trajectory length

        Returns:
            A list of RewardScaleEntry objects
        """
        return cls.generate_reward_scale_entries(
            [
                (
                    90,
                    100,
                    "The reasoning is exceptional, demonstrating comprehensive understanding, thorough analysis of multiple aspects, clear structure, and leads to a highly effective action plan.",
                ),
                (
                    75,
                    89,
                    "The reasoning is very good, showing strong understanding of the problem, logical structure, consideration of alternatives, and leading to a clear action plan.",
                ),
                (
                    50,
                    74,
                    "The reasoning is adequate, showing basic understanding of the problem and some structure, but may lack depth in certain areas or comprehensive consideration of alternatives.",
                ),
                (
                    25,
                    49,
                    "The reasoning shows limited understanding of the problem, lacks structure or depth, and doesn't lead to a clear action plan.",
                ),
                (
                    0,
                    24,
                    "The reasoning is superficial, demonstrates minimal understanding, and fails to address key aspects of the problem.",
                ),
                (
                    -49,
                    -1,
                    "The reasoning is counterproductive, containing significant misconceptions or errors that would lead to incorrect approaches.",
                ),
                (
                    -100,
                    -50,
                    "The reasoning demonstrates fundamental misunderstanding of the problem, contains severe technical errors, or is completely irrelevant to the task at hand.",
                ),
            ]
        )
