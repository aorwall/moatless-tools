import logging
import os
from datetime import time

import openai
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

from ghostcoder import Ghostcoder
from ghostcoder.schema import TextItem

# openai.api_key = st.secrets["OPENAI_API_KEY"]


logging_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logging.getLogger('ghostcoder').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

class StreamHandler(BaseCallbackHandler):

    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


with st.sidebar:
    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    repo_dir = st.text_input("Repository Directory", key="repo_dir", value="/repo")
    model_name = st.text_input("Model name", key="model_name", value=os.environ.get('MODEL_NAME', 'gpt-4'))
    debug_mode = st.toggle('Debug mode', False)

    init_button = st.button("Index repository")

def write_code():
    st.session_state.messages.append({"role": "user", "content": "Write code"})

    with st.chat_message("user"):
        st.markdown("Write code")

    message_placeholder = st.empty()
    stream_handler = StreamHandler(message_placeholder, display_method='write')

    response_message = st.session_state.ghostcoder.write_code(callback=stream_handler)

    response = str(response_message)

    message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


if init_button:
    if repo_dir: # openai_api_key and
        st.session_state.messages = []

        with st.spinner('Indexing repository...'):
            st.session_state.ghostcoder = Ghostcoder(repo_dir=repo_dir, debug_mode=debug_mode, model_name=model_name)

        # Set OpenAI API key
        # openai.api_key = openai_api_key
        st.success("The repository was indexed successfully!")
    else:
        st.error("Please enter Repository Directory.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about anything"):
    if "ghostcoder" not in st.session_state:
        st.error("Please index the repository first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    message_placeholder = st.empty()
    stream_handler = StreamHandler(message_placeholder, display_method='write')

    response = st.session_state.ghostcoder.request(prompt, callback=stream_handler)

    message_placeholder.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
