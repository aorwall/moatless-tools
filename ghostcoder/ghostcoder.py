import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from llama_index.vector_stores import ChromaVectorStore

from ghostcoder.actions.write_code import CodeWriter
from ghostcoder.actions.verify.code_verifier import CodeVerifier
from ghostcoder.actions.write_code.base import StreamCallback
from ghostcoder.actions.write_code.prompt import FIX_TESTS_PROMPT
from ghostcoder.callback import LogCallbackHandler
from ghostcoder.codeblocks import CodeBlockType, create_parser
from ghostcoder.filerepository import FileRepository
from ghostcoder.index.code_index import CodeIndex
from ghostcoder.ipython_callback import DisplayCallback
from ghostcoder.llm import LLMWrapper, ChatLLMWrapper
from ghostcoder.schema import Message, UpdatedFileItem, TextItem, FileItem
from ghostcoder.test_tools import TestTool
from ghostcoder.utils import count_tokens

logger = logging.getLogger(__name__)

def create_openai_client(
        log_dir: Path,
        llm_name: str,
        temperature: float,
        streaming: bool = True,
        openai_api_key: str = None,
        max_tokens: Optional[int] = None,
        stop_sequence: str = None):
    callback = LogCallbackHandler(str(log_dir))
    logger.info(f"create_openai_client(): llm_name={llm_name}, temperature={temperature}, log_dir={log_dir}")

    model_kwargs = {}
    if stop_sequence:
        model_kwargs["stop"] = [stop_sequence]

    return ChatLLMWrapper(ChatOpenAI(
        model=llm_name,
        openai_api_key=openai_api_key,
        model_kwargs=model_kwargs,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
        callbacks=[callback]
    ))


class Ghostcoder:

    def __init__(self,
                 model_name: str = "gpt-4",
                 llm: LLMWrapper = None,
                 basic_llm: LLMWrapper = None,
                 repository: FileRepository = None,
                 code_index: CodeIndex = None,
                 code_writer_sys_prompt: str = None,
                 verify_code: bool = False,
                 test_tool: TestTool = None,
                 auto_mode: bool = True,
                 language: str = None,
                 callback: DisplayCallback = None,
                 max_retries: int = 3,
                 log_dir: str = None,
                 openai_api_key: str = None,
                 index_dir: str = None,
                 repo_dir: str = None,
                 search_limit: int = 5,
                 debug_mode: bool = False):

        exclude_dirs = [".index", ".prompt_log"]

        self.repository = repository or FileRepository(repo_path=Path(repo_dir), exclude_dirs=exclude_dirs)

        repo_dir = repo_dir or repository.repo_path
        log_dir = log_dir or repo_dir + "/.prompt_log"
        log_path = Path(log_dir)
        self.llm = llm or create_openai_client(log_dir=log_path, llm_name=model_name, temperature=0.01, streaming=True, max_tokens=2000, openai_api_key=openai_api_key)
        self.basic_llm = basic_llm or create_openai_client(log_dir=log_path, llm_name="gpt-3.5-turbo", temperature=0.0, streaming=True, openai_api_key=openai_api_key)

        self.index_dir = index_dir or repo_dir + "/.index"
        db = chromadb.PersistentClient(path=self.index_dir + "/.chroma_db")
        chroma_collection = db.get_or_create_collection("code-index")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.code_index = code_index or CodeIndex(repository=self.repository, index_dir=self.index_dir, vector_store=vector_store, limit=search_limit)

        self.code_writer = CodeWriter(llm=self.llm ,
                                      sys_prompt=code_writer_sys_prompt,
                                      repository=self.repository,
                                      callback=callback,
                                      expect_complete_functions=True,
                                      auto_mode=True)

        self.test_fix_writer = CodeWriter(llm=self.llm,
                                          sys_prompt=FIX_TESTS_PROMPT,
                                          repository=self.repository,
                                          callback=callback,
                                          auto_mode=True)
        self.max_retries = max_retries
        self.auto_mode = auto_mode
        self.callback = callback

        self.message_history = []
        self.file_context = []

        if verify_code:
            self.verifier = CodeVerifier(repository=self.repository, test_tool=test_tool, language=language, callback=callback)
        else:
            self.verifier = None

        self.debug_mode = debug_mode
        self.file_context = []

        self.filter_context = False

    def request(self, query: str,
                ability: str = None,
                search_limit: int = 10,
                content_type: str = None,
                callback: BaseCallbackHandler = None) -> str:

        debug_text = ""
        if not ability:

            system_prompt = """Decide on what ability to use based on the users input. Return only the name of the ability
    
    ## Abilities
    You have access to the following abilities you can call:
    - write_code: Write code.
    - investigate: Investigate and answer questions from the user.
    """
            message = Message(sender="User", items=[TextItem(text=query)])

            ability, _ = self.basic_llm.generate(system_prompt, messages=[message])

            if self.debug_mode:
                debug_text = f"> _Selected ability: {ability}_\n\n"
                callback.on_llm_new_token(debug_text)

            logger.info(debug_text)

        if "write_code" in ability:
            if not self.file_context:
                return "Please provide at least one file in the context."
            return debug_text + self.write_code(query, callback)
        else:
            return debug_text + self.investigate(query,
                                                 content_type=content_type,
                                                 search_limit=search_limit,
                                                 callback=callback)

    def write_code(self, query: str, callback: BaseCallbackHandler = None) -> str:
        logger.info(f"Write code")

        debug_text_before = f"> _Provided files from file context:_"
        for i, file_item in enumerate(self.file_context):
             debug_text_before += f"\n> {i + 1}. `{file_item.file_path}`"

        debug_text_before += "\n\n"

        if self.debug_mode:
            callback.on_llm_new_token(debug_text_before)

        incoming_messages = []
        incoming_messages.append(Message(sender="User", items=self.file_context))
        incoming_messages.extend(self.message_history)
        incoming_messages.append(Message(sender="User", items=[TextItem(text=query)]))

        outgoing_messages = self.code_writer.execute(incoming_messages=incoming_messages, callback=callback)

        if self.debug_mode:
            return debug_text_before + str(outgoing_messages[-1])

        return str(outgoing_messages[-1])

    def find_files(self, incoming_message: Message) -> List[Message]:
        if not self.code_index:
            raise Exception("Code index not initialized.")

        query = incoming_message.to_prompt()
        hits = self.code_index.search(query)

        for hit in hits:
            print(hit.path)
            for codeblock in hit.blocks:
                print(codeblock.score)
                print(codeblock.type)
                print(codeblock.content)

        # TODO: Run code index and pick files
        return []

    def investigate(self, query: str,
                    search_limit: int = 10,
                    content_type: str = None,
                    callback: BaseCallbackHandler = None):
        message = Message(sender="User", items=[TextItem(text=query)])
        self.message_history.append(message)

        hits = self.code_index.search(str(self.message_history), content_type=content_type, limit=search_limit)

        debug_text_before = f"> _Vector store search hits : {len(hits)}_"
        for i, hit in enumerate(hits):
            if self.debug_mode:
                debug_text_before += f"\n> {i+1}. `{hit.path}` ({len(hit.blocks)} blocks "
                debug_text_before += ", ".join([f"`{block.identifier}`" for block in hit.blocks])
                debug_text_before += ")"

            if any([hit.path in item.file_path for item in self.file_context]):
                continue

            content = self.repository.get_file_content(file_path=hit.path)
            self.file_context.append(FileItem(file_path=hit.path, content=content))

        logger.debug(debug_text_before)

        debug_text_before += "\n\n"

        if self.debug_mode:
            callback.on_llm_new_token(debug_text_before)

        system_prompt = """You're an AI developer with superior programming skills. 

You're task is to help a non technical person to understand a bug in a large codebase. Try to make short but informative responses.
You are provided with a list of files that might be relevant to the question. If they aren't relevant you can just ignore them. 

YOU MUST provide the full file path to files you refer to.

DO NOT suggest code changes or show hypothetical example in code that is not in the context.
You can ask to read files not in context by providing the full file path to the file.

When you return code you should use the following format:

/file.py
```python
# ... code  
```
"""

        file_context_message = Message(sender="User", items=self.file_context)

        file_tree_message = Message(sender="User", items=[TextItem(text="Existing files:\n" + self.repository.file_tree().tree_string(content_type="code"))])

        response, _ = self.llm.generate(system_prompt, messages=[file_context_message, file_tree_message] + self.message_history, callback=callback)

        debug_text_after = f"\n\n> _Filtered context files:_"

        if self.filter_context:
            filtered_context = []
            i = 0
            for file_item in self.file_context:
                if self.is_mentioned(file_item, response):
                    i += 1
                    debug_text_after += f"\n> {i}. `{file_item.file_path}`"
                    filtered_context.append(file_item)
            self.file_context = filtered_context

        found_files = self.repository.find_files_in_content(content=response)
        new_file_count = 0
        for found_file in found_files:
            if any([found_file.path in item.file_path for item in self.file_context]):
                logger.debug(f"File {found_file.path} already in context.")
                continue

            if new_file_count == 0:
                debug_text_after = f"\n>\n> _Files mentioned in message:_"

            new_file_count += 1
            debug_text_after += f"\n> {new_file_count}. `{found_file.path }`"
            content = self.repository.get_file_content(file_path=found_file.path)

            self.file_context.append(FileItem(file_path=found_file.path, content=content))

        if not self.file_context:
            debug_text_after = "\n\n> _No filtered files in context._\n"

        logger.debug(debug_text_after)
        if self.debug_mode:
            callback.on_llm_new_token(debug_text_after)

        message = Message(sender="AI", items=[TextItem(text=response)])
        self.message_history.append(message)

        if self.debug_mode:
            response = debug_text_before + response + debug_text_after

        return response

    def is_mentioned(self, file_item, response):
        if file_item.file_path.split("/")[-1] in response:
            return True

        try:
            code_block = create_parser(file_item.language).parse(file_item.content)
            child_blocks = code_block.find_blocks_with_types([CodeBlockType.FUNCTION, CodeBlockType.CLASS])
            for child_block in child_blocks:
                if child_block.identifier and child_block.identifier in response:
                    return True
        except:
            logger.info(f"Failed to parse file {file_item.file_path}")

        return False


    def run(self, message: Message) -> List[Message]:
        file_items = message.find_items_by_type(item_type="file")
        for file_item in file_items:
            if not file_item.content and self.repository:
                logger.debug(f"Get current file content for {file_item.file_path}")
                content = self.repository.get_file_content(file_path=file_item.file_path)
                if content:
                    file_item.content = content
                elif file_item.new:
                    file_item.content = ""
                else:
                    raise Exception(f"File {file_item.file_path} not found in repository.")

        if self.callback:
            self.callback.display_message(message)

        return self._run(message)

    def _run(self, incoming_message: Message) -> List[Message]:
        outgoing_messages = self.code_writer.execute(incoming_messages=[incoming_message])
        outgoing_messages.extend(self.verify(messages=[incoming_message] + outgoing_messages))
        return outgoing_messages

    def verify(self, messages: List[Message], retry: int = 0, last_run: int = 0) -> [Message]:
        if not self.verifier:
            return []

        updated_files = dict()

        for message in messages:
            for item in message.items:
                if isinstance(item, UpdatedFileItem) and not item.invalid:
                    updated_files[item.file_path] = item

        if not updated_files:
            # TODO: Handle if no files where updated in last run?
            return []

        file_items = []
        for file_item in updated_files.values():
            if self.repository:
                content = self.repository.get_file_content(file_path=file_item.file_path)
            else:
                content = file_item.content

            file_items.append(FileItem(file_path=file_item.file_path,
                                       content=content,
                                       invalid=file_item.invalid))

        outgoing_messages = []

        logger.info(f"Updated files, verifying...")
        verification_message = self.verifier.execute()
        if self.callback:
            self.callback.display_message(verification_message)
        outgoing_messages.append(verification_message)

        failures = verification_message.find_items_by_type("verification_failure")
        if failures:
            if retry < self.max_retries or len(failures) < last_run:
                verification_message.items.extend(file_items)

                retry += 1
                incoming_messages = self.make_summary(messages)

                logger.info(f"{len(failures)} verifications failed (last run {last_run}, retrying ({retry}/{self.max_retries})...")
                incoming_messages.append(verification_message)
                response_messages = self.test_fix_writer.execute(incoming_messages=incoming_messages)
                return self.verify(messages=messages + [verification_message] + response_messages,
                                   retry=retry,
                                   last_run=len(failures))
            else:
                logger.info(f"Verification failed, giving up...")

        return outgoing_messages

    def make_summary(self, messages: List[Message]) -> List[Message]:
        summarized_messages = []
        sys_prompt = """Make a short summary of the provided message."""

        for message in messages:
            if message.sender == "Human":
                text_items = message.find_items_by_type("text")
                summarized_messages.append(Message(sender=message.sender, items=text_items))
            else:
                if not message.summary and self.basic_llm:
                    message.summary, stats = self.basic_llm.generate(sys_prompt, messages=[message])
                    logger.debug(f"Created summary {stats.json}")
                if message.summary:
                    summarized_messages.append(Message(sender=message.sender, items=[TextItem(text=message.summary)]))

        return summarized_messages

    def get_file_context(self):
        files = [file.file_path for file in self.file_context]
        files.sort()
        return files

    def add_file_to_context(self, file_path):
        content = self.repository.get_file_content(file_path=file_path)
        if content:
            self.file_context.append(FileItem(file_path=file_path, content=content))

    def remove_file_from_context(self, file_path):
        self.file_context = [file for file in self.file_context if file.file_path != file_path]

    def get_file_context_tokens(self):
        tokens = [count_tokens(file.content) for file in self.file_context]
        return sum(tokens)