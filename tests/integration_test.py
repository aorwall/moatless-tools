import os

import pytest
from dotenv import load_dotenv

from moatless.agent.code_agent import CodingAgent
from moatless.model_config import SUPPORTED_MODELS
from moatless.benchmark.swebench import load_instance, create_repository
from moatless.completion.base import BaseCompletionModel
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)

@pytest.mark.parametrize(
    "model_config",
    SUPPORTED_MODELS,
    ids=[config["model"].replace("/", "_") for config in SUPPORTED_MODELS]
)
@pytest.mark.llm_integration
def test_basic_coding_task(model_config):
    completion_model = BaseCompletionModel(
        model=model_config["model"],
        temperature=model_config["temperature"],
        response_format=model_config["response_format"],
        thoughts_in_action=model_config["thoughts_in_action"],
        disable_thoughts=model_config["disable_thoughts"]
    )

    instance = load_instance("django__django-16527")
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
        disable_thoughts=model_config["disable_thoughts"]
    )

    persist_path = f"integration_test_{model_config['model'].replace('.', '_').replace('/', '_')}.json"

    loop = AgenticLoop.create(
        f"<task>\n{instance['problem_statement']}\n</task>",
        agent=agent,
        repository=repository,
        max_iterations=15,
        persist_path=persist_path
    )

    loop.maybe_persist()
    node = loop.run()
    print(node.message)
    usage = loop.total_usage()
    print(usage)
    loop.maybe_persist()
    assert node.action
    assert node.action.name == "Finish"
    assert loop.is_finished()

