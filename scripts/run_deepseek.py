
import logging
import os

from dotenv import load_dotenv
from moatless.benchmark.swebench import create_repository
from moatless.benchmark.utils import get_moatless_instance
from moatless.agent.code_agent import CodingAgent
from moatless.index import CodeIndex
from moatless.loop import AgenticLoop
from moatless.file_context import FileContext
from moatless.completion.base import LLMResponseFormat
from moatless.schema import MessageHistoryType

load_dotenv()

index_store_dir = os.getenv("INDEX_STORE_DIR", "/tmp/index_store")
repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

persist_path = "trajectory.json"

logging.basicConfig(level=logging.INFO)

instance = get_moatless_instance("django__django-16379")
repository = create_repository(instance)
code_index = CodeIndex.from_index_name(
    instance["instance_id"], 
    index_store_dir=index_store_dir, 
    file_repo=repository
)
file_context = FileContext(repo=repository)

# Create agent using Deepseek Chat with explicit config
agent = CodingAgent.create(
    repository=repository,
    code_index=code_index,
    
    model="deepseek/deepseek-chat",
    temperature=0.0,
    max_tokens=4000,
    few_shot_examples=True,
    
    response_format=LLMResponseFormat.REACT,
    message_history_type=MessageHistoryType.REACT
)

loop = AgenticLoop.create(
    message=instance["problem_statement"],
    agent=agent,
    file_context=file_context,
    repository=repository,
    persist_path=persist_path,
    max_iterations=50,
    max_cost=2.0
)

final_node = loop.run()
if final_node:
    print(final_node.observation.message)