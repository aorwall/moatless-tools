#!/usr/bin/env python3
"""Script to run simple flow integration tests."""

import os

from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import create_repository
from moatless.completion.base import BaseCompletionModel
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.validation.base_code_flow_validation import BaseCodeFlowValidation


class SimpleFlowValidation(BaseCodeFlowValidation):
    def validate_result(self, node, loop) -> bool:
        """Validate that the loop finished successfully and produced a patch."""
        return node.action and node.action.name == "Finish" and node.file_context.has_patch()

    def create_loop(self, model_config: dict, instance: dict) -> AgenticLoop:
        completion_model = BaseCompletionModel.create(
            model=model_config["model"],
            temperature=model_config["temperature"],
            response_format=model_config["response_format"],
            thoughts_in_action=model_config["thoughts_in_action"],
        )

        repository = create_repository(instance)

        index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
        code_index = CodeIndex.from_index_name(
            instance["instance_id"], index_store_dir=index_store_dir, file_repo=repository
        )

        agent = CodingAgent.create(
            completion_model=completion_model,
            repository=repository,
            code_index=code_index,
            message_history_type=model_config["message_history_type"],
            thoughts_in_action=model_config["thoughts_in_action"],
        )

        return AgenticLoop.create(
            f"<task>\n{instance['problem_statement']}\n</task>", agent=agent, repository=repository, max_iterations=15
        )


if __name__ == "__main__":
    SimpleFlowValidation().run()
