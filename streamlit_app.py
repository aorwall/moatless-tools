import logging
import os
from datetime import time

import openai
import streamlit as st

from ghostcoder.display_callback import DisplayCallback
from ghostcoder.runtime.openaichat import OpenAIChat
from ghostcoder.schema import TextItem, Item, FunctionItem, Message

# openai.api_key = st.secrets["OPENAI_API_KEY"]

logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logging.getLogger('ghostcoder').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class CallbackHandler(DisplayCallback):

    def __init__(self, container, display_method='markdown'):
        self.container = container
        self.items = []
        self.display_method = display_method

    def on_new_item(self, new_item: Item):
        logger.info("on_new_item")
        self.items.append(new_item)

        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            if isinstance(new_item, TextItem):
                display_function(new_item.text)
            elif isinstance(new_item, FunctionItem):
                display_function(new_item.dict())
        else:
            logger.error(f"Invalid display_method: {self.display_method}")


with st.sidebar:
    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    repo_dir = st.text_input("Repository Directory", key="repo_dir", value=os.environ.get('REPO_DIR', ''))
    model_name = st.text_input("Model name", key="model_name", value=os.environ.get('MODEL_NAME', 'gpt-4-1106-preview'))
    debug_mode = st.toggle('Debug mode', True)

    if "ghostcoder" not in st.session_state:
        init_button = st.button("Load")
    else:
        init_button = st.button("Reload")


if init_button:
    if repo_dir:  # openai_api_key and
        st.session_state.messages = []

        with st.spinner('Indexing repository...'):
            st.session_state.ghostcoder = OpenAIChat(model_name=model_name, repo_dir=repo_dir, debug_mode=debug_mode)

        # Set OpenAI API key
        # openai.api_key = openai_api_key
        st.success("The repository was indexed successfully!")
        st.rerun()
    else:
        st.error("Please enter Repository Directory.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.role):
        for item in message.items:
            if isinstance(item, TextItem):
                st.markdown(item.text)
            elif isinstance(item, FunctionItem):
                header = f"{item.name}("
                for i, arg in enumerate(item.arguments):
                    if i > 0:
                        header += ", "
                    header += f"{arg}"
                header += ")"

                with st.expander(header):
                    if item.output:
                        st.text("Output")
                        st.json(item.output)

if prompt := st.chat_input("Ask about anything"):
    if "ghostcoder" not in st.session_state:
        st.error("Please index the repository first.")
        st.stop()

    message = Message(role="user", items=[TextItem(text=prompt)])
    st.session_state.messages.append(message)
    with st.chat_message("user"):
        st.markdown(prompt)

    message_placeholder = st.empty()
    callback = CallbackHandler(message_placeholder, display_method='write')

    with st.spinner('Wait for it...'):
        response_message = st.session_state.ghostcoder.send(prompt, callback=callback)
        st.session_state.messages.append(response_message)

    st.rerun()