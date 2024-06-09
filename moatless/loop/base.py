import logging
from abc import abstractmethod, ABC
from typing import Optional, Callable, Type, Any

import litellm
from pydantic import BaseModel

from moatless.llm.completion import completion
from moatless.settings import Settings
from moatless.trajectory import Trajectory
from moatless.types import ActionResponse, ActionSpec

logger = logging.getLogger(__name__)


class LoopState(BaseModel):

    def model(self) -> str:
        return Settings.agent_model

    def temperature(self) -> float:
        return 0.0

    def tools(self) -> list[Type[ActionSpec]]:
        return []

    def stop_words(self):
        return []

    def max_tokens(self):
        return 1000


class NoState(LoopState):
    pass


class BaseLoop(ABC):
    def __init__(
        self,
        completion_mock: Optional[Callable] = None,
        trajectory: Optional[Trajectory] = None,
        max_cost: float = 0.25,
    ):
        """
        Initialize the Loop instance.

        Args:
            model (Optional[str]): Optional LLM model name.
            completion_mock (Optional[Callable]): Optional mock function for completion.
            trajectory (Optional[Trajectory]): Optional trajectory instance.
            max_cost (float): Maximum allowed cost for the loop execution.
        """

        self._state: LoopState = NoState()
        self._completion = completion_mock or completion
        self._trajectory = trajectory
        self._max_cost = max_cost
        self._is_running = False

    def execute(self) -> ActionResponse:
        """
        Execute the loop until completion or until the maximum cost is exceeded.
        """
        if self._is_running:
            raise Exception("The code finder is already running.")

        self._is_running = True
        try:
            response = self._run_loop()
            return response
        finally:
            self._is_running = False

    def _run_loop(self) -> ActionResponse:
        """
        Run the loop and handle exceptions and cost checking.
        """
        while self._is_running:
            try:
                response_message = self._run_completion()
                logger.info(
                    f"Received message. Current state: {self.state.__class__.__name__}"
                )
                response = self.loop(response_message)
                if response:
                    return response
            except Exception as e:
                logger.warning(f"Failed to run loop. Error: {e}")
                raise

            if self._check_cost_exceeded():
                raise Exception(
                    "The search was aborted because the cost exceeded the limit."
                )

        raise Exception("Loop exited without returning a response.")

    def _check_cost_exceeded(self) -> bool:
        """
        Check if the total cost has exceeded the maximum allowed cost.

        Returns:
            bool: True if the cost has exceeded, False otherwise.
        """
        if not self._trajectory:
            return False

        total_cost = self._trajectory.total_cost()
        if total_cost > self._max_cost:
            logger.warning(f"Max cost reached ({self._max_cost}). Exiting.")
            self._trajectory.save_error(
                f"Total cost {total_cost} exceeded max allowed cost {self._max_cost}."
            )
            self._is_running = False
            return True
        return False

    @abstractmethod
    def loop(self, message: litellm.Message) -> Optional[ActionResponse]:
        """
        Abstract method to be implemented by subclasses to define the loop logic.
        Will be run until a response is returned.

        Args:
            message (litellm.Message): Last response from the LLM

        Returns:
            Optional[ActionResponse]: The response from the loop, if any.
        """
        raise NotImplementedError

    @abstractmethod
    def message_history(self) -> list[dict]:
        raise NotImplementedError

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [tool.openai_tool_spec() for tool in self.state.tools()]

    @property
    def state(self):
        return self._state

    def transition(self, new_state: LoopState):
        logger.info(
            f"Transitioning from {self.state.__class__.__name__} to {new_state.__class__.__name__}"
        )
        self._state = new_state

    def _run_completion(self) -> litellm.Message:
        if self.state is not NoState:
            generation_name = (
                self.__class__.__name__ + "_" + self.state.__class__.__name__
            )
        else:
            generation_name = self.__class__.__name__

        response = self._completion(
            model=self.state.model(),
            max_tokens=self.state.max_tokens(),
            temperature=self.state.temperature(),
            stop=self.state.stop_words(),
            tools=self._tool_specs(),
            generation_name=generation_name,
            messages=self.message_history(),  # TODO: Should be from state also
        )

        if self._trajectory:
            self._trajectory.new_step(
                name=generation_name, completion_response=response
            )

        return response.choices[0].message
