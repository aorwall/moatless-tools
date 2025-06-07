import asyncio
import logging
import os
from dotenv import load_dotenv
import litellm
from moatless.actions import Respond
from moatless.agent import ActionAgent
from moatless.completion.log_handler import LogHandler
from moatless.completion.manager import ModelConfigManager
from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.message_history.message_history import MessageHistoryGenerator
from moatless.storage.file_storage import FileStorage
from moatless.context_data import current_project_id, current_trajectory_id
from moatless.node import Node

load_dotenv()


async def smoke_test(model_id: str):
    storage = FileStorage(base_dir=os.getenv("MOATLESS_DIR"))
    model_config_manager = ModelConfigManager(storage=storage)
    await model_config_manager.initialize()
    current_project_id.set("smoke_test")
    current_trajectory_id.set("smoke_test")
    litellm.callbacks = [LogHandler(storage=storage)]

    completion_model = model_config_manager.get_model_config(model_id)

    agent = ActionAgent(
        completion_model=completion_model,
        system_prompt="You are a helpful assistant that can answer questions.",
        actions=[
            Respond()
        ],
    )

    node = Node.create_root(user_message="Hello")
    node = node.create_child(user_message="Hello")
    await agent.run(node)
    print(node.observation.message)
    
    node = node.create_child(user_message="What is the capital of France?")
    await agent.run(node)
    print(node.observation.message)
    
    node = node.create_child(user_message="Goodbye")
    await agent.run(node)
    print(node.observation.message)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(smoke_test("claude-sonnet-4-20250514-thinking"))