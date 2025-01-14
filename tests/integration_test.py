import os

import pytest
from dotenv import load_dotenv

from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import load_instance, create_repository
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.schema import MessageHistoryType
from moatless.search_tree import SearchTree

load_dotenv()
moatless_dir = os.getenv("MOATLESS_DIR", "/tmp/moatless")

global_params = {
    "model": "gpt-4o-mini-2024-07-18",
    "temperature": 0.5,
    "max_tokens": 2000,
    "max_prompt_file_tokens": 8000,
}

pytest.mark.llm_integration = pytest.mark.skipif(
    "not config.getoption('--run-llm-integration')",
    reason="need --run-llm-integration option to run tests that call LLMs",
)


@pytest.mark.parametrize(
    "model",
    [
        # Claude 3.5 Sonnet
        {
            "model": "claude-3-5-sonnet-20241022",
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
            "thoughts_in_action": False
        },
        # Claude 3.5 Haiku
        {
            "model": "claude-3-5-haiku-20241022",
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
            "thoughts_in_action": False
        },
        # GPT-4o
        {
            "model": "azure/gpt-4o",
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
            "thoughts_in_action": True
        },
        # GPT-4o Mini
        {
            "model": "azure/gpt-4o-mini",
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
            "thoughts_in_action": True
        },
        # o1 preview
        {
            "model": "o1-preview-2024-09-12",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": False
        },
        # o1 Mini
        {
            "model": "o1-mini-2024-09-12",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": False
        },
        # DeepSeek Chat
        {
            "model": "deepseek/deepseek-chat",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": True
        },
        # Gemini Flash
        {
            "model": "gemini/gemini-2.0-flash-exp",
            "response_format": LLMResponseFormat.TOOLS,
            "message_history_type": MessageHistoryType.MESSAGES,
            "thoughts_in_action": True
        },
        # Gemini Flash Think
        {
            "model": "gemini/gemini-2.0-flash-thinking-exp",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": True
        },
        # Llama 3.1 70B Instruct
        {
            "model": "openrouter/meta-llama/llama-3.1-70b-instruct",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": False
        },
        # Qwen 2.5 Coder
        {
            "model": "openrouter/qwen/qwen-2.5-coder-32b-instruct",
            "response_format": LLMResponseFormat.REACT,
            "message_history_type": MessageHistoryType.REACT,
            "thoughts_in_action": False
        }
    ],
    ids=["claude-3-5-sonnet", "claude-3-5-haiku", "gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "deepseek-chat", "gemini-2.0-flash", "gemini-2.0-flash-think", "llama-3.1-70b", "qwen-2.5-coder"]
)
@pytest.mark.llm_integration
def test_basic_coding_task(model):
    completion_model = CompletionModel(
        model=model["model"], 
        temperature=0.0, 
        response_format=model["response_format"], 
        thoughts_in_action=model["thoughts_in_action"]
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
        message_history_type=model["message_history_type"],
        thoughts_in_action=model["thoughts_in_action"]
    )

    persist_path = f"itegration_test_{model['model'].replace('.', '_').replace('/', '_')}.json"

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
    loop.maybe_persist()
    assert node.action
    assert node.action.name == "Finish"
    assert loop.is_finished()
    # print(json.dumps(search_tree.root.model_dump(), indent=2))

