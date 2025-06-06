import asyncio
import logging
import mimetypes
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import aiofiles
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.storage.docstore import DocumentStore, SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from opentelemetry import trace
from rapidfuzz import fuzz

from moatless.codeblocks import CodeBlock, CodeBlockType, get_parser_by_path
from moatless.codeblocks.module import Module
from moatless.index.code_block_index import CodeBlockIndex
from moatless.index.embed_model import get_embed_model
from moatless.index.settings import IndexSettings
from moatless.index.simple_faiss import SimpleFaissVectorStore
from moatless.index.types import (
    CodeSnippet,
    SearchCodeHit,
    SearchCodeResponse,
)
from moatless.repository import FileRepository
from moatless.repository.repository import Repository
from moatless.utils.file import is_test
from moatless.utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

# Add constant for persist filename outside TYPE_CHECKING
DEFAULT_PERSIST_FNAME = "docstore.json"


def default_vector_store(settings: IndexSettings):
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss needs to be installed to set up a default index for CodeIndex. Run 'pip install faiss-cpu'"
        ) from e

    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(settings.dimensions))
    from moatless.index.simple_faiss import SimpleFaissVectorStore

    return SimpleFaissVectorStore(faiss_index)


class CodeIndex:
    def __init__(
        self,
        file_repo: Repository | None = None,
        index_name: Optional[str] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        docstore: Optional[DocumentStore] = None,
        embed_model: Optional[BaseEmbedding] = None,
        code_block_index: Optional[CodeBlockIndex] = None,
        settings: Optional[IndexSettings] = None,
        max_results: int = 25,
        max_hits_without_exact_match: int = 100,
        max_exact_results: int = 5,
    ):
        self._index_name = index_name
        self._settings = settings or IndexSettings()

        self.max_results = max_results
        self.max_hits_without_exact_match = max_hits_without_exact_match
        self.max_exact_results = max_exact_results

        self._file_repo = file_repo

        self._code_block_index = code_block_index

        self._embed_model = embed_model or get_embed_model(self._settings.embed_model)
        self._vector_store = vector_store or default_vector_store(self._settings)
        self._docstore = docstore or SimpleDocumentStore()

        self._thread_pool = ThreadPoolExecutor(max_workers=5)

    async def matching_files(self, file_pattern: str) -> list[str]:
        """
        Returns a list of files matching the given pattern within the repository.
        Uses inverted indexes instead of filesystem operations.

        Parameters:
            file_pattern (str): The glob pattern to match files.

        Returns:
            List[str]: A list of relative file paths matching the pattern.
        """
        try:
            # If absolute path, log warning and remove first slash
            if file_pattern.startswith("/"):
                logger.warning(f"Converting absolute path {file_pattern} to relative path")
                file_pattern = file_pattern[1:]

            matched_files = sorted(await self._code_block_index.match_glob_pattern(file_pattern))
            return matched_files

        except Exception:
            logger.exception(f"Error finding files for pattern {file_pattern}:")
            return []

    async def find_by_pattern(self, patterns: list[str]) -> list[str]:
        """
        Returns a list of files matching the given patterns within the repository.
        Uses inverted indexes instead of filesystem operations.
        """
        matched_files = set()
        for pattern in patterns:
            matches = await self._code_block_index.match_glob_pattern(f"**/{pattern}")
            matched_files.update(matches)

        return sorted(matched_files)

    @classmethod
    def from_persist_dir(cls, persist_dir: str, file_repo: Repository | None = None, **kwargs):
        """Synchronous version of from_persist_dir"""

        vector_store = SimpleFaissVectorStore.from_persist_dir(persist_dir)

        docstore, settings = (
            SimpleDocumentStore.from_persist_dir(persist_dir),
            IndexSettings.from_persist_dir(persist_dir),
        )

        code_block_index = CodeBlockIndex.from_persist_dir(persist_dir)

        return cls(
            file_repo=file_repo,
            vector_store=vector_store,
            docstore=docstore,
            settings=settings,
            code_block_index=code_block_index,
            **kwargs,
        )

    @classmethod
    async def from_persist_dir_async(cls, persist_dir: str, file_repo: Repository | None = None, **kwargs):
        """Asynchronous version of from_persist_dir"""
        # Run CPU-intensive synchronous operations in a thread pool
        loop = asyncio.get_event_loop()

        # Use the async version of SimpleFaissVectorStore.from_persist_dir
        vector_store = await SimpleFaissVectorStore.from_persist_dir_async(persist_dir)

        from llama_index.core.storage.docstore import SimpleDocumentStore

        # These are still synchronous operations
        docstore, settings = await asyncio.gather(
            loop.run_in_executor(None, SimpleDocumentStore.from_persist_dir, persist_dir),
            loop.run_in_executor(None, IndexSettings.from_persist_dir, persist_dir),
        )

        inverted_index = await CodeBlockIndex.from_persist_dir_async(persist_dir)

        return cls(
            file_repo=file_repo,
            vector_store=vector_store,
            docstore=docstore,
            settings=settings,
            code_block_index=inverted_index,
            **kwargs,
        )

    @classmethod
    async def from_url_async(cls, url: str, persist_dir: str, file_repo: FileRepository):
        """Asynchronous version of from_url"""
        import asyncio

        import aiohttp
        import ssl

        # Create a custom SSL context that doesn't verify certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=ssl_context) as response:
                    response.raise_for_status()

                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_zip_file = os.path.join(temp_dir, url.split("/")[-1])

                        # Download file in chunks
                        async with aiofiles.open(temp_zip_file, "wb") as data:
                            async for chunk in response.content.iter_chunked(8192):
                                await data.write(chunk)

                        # Run unpack_archive in thread pool since it's synchronous
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, shutil.unpack_archive, temp_zip_file, persist_dir)

        except aiohttp.ClientError as e:
            logger.exception(f"HTTP Error while fetching {url}")
            raise e
        except Exception as e:
            logger.exception(f"Failed to download {url}")
            raise e

        logger.info(f"Downloaded existing index from {url}.")
        return await cls.from_persist_dir_async(persist_dir, file_repo)

    @classmethod
    async def from_index_name_async(
        cls,
        index_name: str,
        file_repo: Repository,
        index_store_dir: Optional[str] = None,
    ):
        """Asynchronous version of from_index_name"""
        if not index_store_dir:
            index_store_dir = os.getenv("INDEX_STORE_DIR")

        persist_dir = os.path.join(index_store_dir, index_name)
        if os.path.exists(persist_dir):
            logger.info(f"Loading existing index {index_name} from {persist_dir}.")
            return await cls.from_persist_dir_async(persist_dir, file_repo=file_repo)
        else:
            logger.info(f"No existing index found at {persist_dir}.")

        if os.getenv("INDEX_STORE_URL"):
            index_store_url = os.getenv("INDEX_STORE_URL")
        else:
            index_store_url = "https://stmoatless.blob.core.windows.net/indexstore/20250118-voyage-code-3/"

        store_url = os.path.join(index_store_url, f"{index_name}.zip")
        logger.info(f"Downloading existing index {index_name} from {store_url}.")
        return await cls.from_url_async(store_url, persist_dir, file_repo)

    def dict(self):
        return {"index_name": self._index_name}

    async def get_module(self, file_path: str, content: str) -> Module | None:
        parser = get_parser_by_path(file_path)
        if parser:
            module = parser.pars(content)
            return module
        return None

    @tracer.start_as_current_span("semantic_search")
    async def semantic_search(
        self,
        query: Optional[str] = None,
        code_snippet: Optional[str] = None,
        file_pattern: Optional[str] = None,
        category: str | None = None,
        max_results: int = 100,
        max_tokens: int = 8000,
        max_hits_without_exact_match: int = 100,
        max_exact_results: int = 5,
        max_spans_per_file: Optional[int] = None,
        exact_match_if_possible: bool = False,
    ) -> SearchCodeResponse:
        if query is None:
            query = ""

        message = ""
        if file_pattern:
            try:
                matching_files = await self.matching_files(file_pattern)
                matching_files = [file for file in matching_files if not is_test(file)]
            except Exception:
                return SearchCodeResponse(
                    message=f"The file pattern {file_pattern} is invalid.",
                    hits=[],
                )

            if not matching_files:
                if "*" not in file_pattern and not self._file_repo.file_exists(file_pattern):
                    return SearchCodeResponse(
                        message=f"No file found on path {file_pattern}.",
                        hits=[],
                    )
                else:
                    return SearchCodeResponse(
                        message=f"No files found for file pattern {file_pattern}.",
                        hits=[],
                    )

        search_results = await self._vector_search(
            query,
            file_pattern=file_pattern,
            exact_content_match=code_snippet,
            category=category,
        )

        files_with_spans: dict[str, SearchCodeHit] = {}

        span_count = 0
        spans_with_exact_query_match = 0
        filtered_out = 0

        require_exact_query_match = False

        sum_tokens = 0
        for rank, search_hit in enumerate(search_results):
            file = self._file_repo.get_file(search_hit.file_path)
            if not file:
                logger.warning(
                    f"semantic_search(query={query}, file_pattern={file_pattern}) Could not find search hit file {search_hit.file_path}."
                )
                continue
            elif not file.module:
                logger.warning(
                    f"semantic_search(query={query}, file_pattern={file_pattern}) Could not parse module for search hit file {search_hit.file_path}."
                )
                continue

            # TODO: Add a check before span is added...
            if sum_tokens > max_tokens:
                break

            spans = []
            for span_id in search_hit.span_ids:
                span = file.module.find_span_by_id(span_id)

                if span:
                    spans.append(span)
                else:
                    logger.debug(f"semantic_search() Could not find span with id {span_id} in file {file.file_path}")

                    spans_by_line_number = file.module.find_spans_by_line_numbers(
                        search_hit.start_line, search_hit.end_line
                    )

                    for span_by_line_number in spans_by_line_number:
                        spans.append(span_by_line_number)

            for span in spans:
                has_exact_query_match = (
                    exact_match_if_possible and query and span.initiating_block.has_content(query, span.span_id)
                )

                if code_snippet and not span.initiating_block.has_content(code_snippet, span.span_id):
                    filtered_out += 1
                    continue

                if has_exact_query_match:
                    spans_with_exact_query_match += 1

                if has_exact_query_match and not require_exact_query_match:
                    require_exact_query_match = True
                    files_with_spans = {}

                if (not require_exact_query_match and span_count <= max_results) or has_exact_query_match:
                    if search_hit.file_path not in files_with_spans:
                        files_with_spans[search_hit.file_path] = SearchCodeHit(file_path=search_hit.file_path)

                    if files_with_spans[search_hit.file_path].contains_span(span.span_id):
                        continue

                    span_count += 1
                    logger.debug(
                        f"semantic_search() Found span {span.span_id} and rank {rank} in file {search_hit.file_path} tokens {sum_tokens+span.tokens}."
                    )

                    files_with_spans[search_hit.file_path].add_span(span_id=span.span_id, rank=rank, tokens=span.tokens)

                    sum_tokens += span.tokens

                    if max_spans_per_file and len(files_with_spans[search_hit.file_path].spans) >= max_spans_per_file:
                        break

            if exact_match_if_possible:
                if spans_with_exact_query_match > max_exact_results or (
                    spans_with_exact_query_match == 0 and span_count > max_hits_without_exact_match
                ):
                    break
            elif span_count > max_results:
                break

        span_count = sum([len(file.spans) for file in files_with_spans.values()])

        if require_exact_query_match:
            logger.info(
                f"semantic_search() Found {spans_with_exact_query_match} code spans with exact match out of {span_count} spans."
            )
            message = (
                f"Found {spans_with_exact_query_match} code spans with code that matches the exact query `{query}`."
            )
        else:
            logger.info(f"semantic_search() Found {span_count} code spans in {len(files_with_spans.values())} files.")
            message = f"Found {span_count} code spans."

        return SearchCodeResponse(message=message, hits=list(files_with_spans.values()))

    @tracer.start_as_current_span("find_class")
    async def find_class(self, class_name: str, file_pattern: Optional[str] = None):
        return await self.find_by_name(class_name=class_name, file_pattern=file_pattern, strict=True)

    @tracer.start_as_current_span("find_function")
    async def find_function(
        self,
        function_name: str,
        class_name: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ):
        return await self.find_by_name(
            function_name=function_name,
            class_name=class_name,
            file_pattern=file_pattern,
            strict=True,
        )

    async def find_by_name(
        self,
        class_name: str = None,
        function_name: str = None,
        file_pattern: Optional[str] = None,
        include_functions_in_class: bool = True,
        strict: bool = False,
        category: str | None = None,
    ) -> SearchCodeResponse:
        if not class_name and not function_name:
            raise ValueError("At least one of class_name or function_name must be provided.")

        paths = []

        # If class name is provided only find the clasees and then filter on function name if necessary
        if class_name:
            paths = await self._code_block_index.get_blocks_by_class(class_name)
        elif function_name:
            paths = await self._code_block_index.get_blocks_by_function(function_name)
        else:
            raise ValueError("At least one of class_name or function_name must be provided.")

        logger.info(f"find_by_name() Found {len(paths)} paths.")

        if file_pattern:
            include_files = await self.matching_files(file_pattern)

            if include_files:
                filtered_paths = []
                for file_path, block_path in paths:
                    if file_path in include_files:
                        filtered_paths.append((file_path, block_path))

                filtered_out_by_file_pattern = len(paths) - len(filtered_paths)
                if filtered_out_by_file_pattern:
                    logger.debug(f"find_by_name() Filtered out {filtered_out_by_file_pattern} files by file pattern.")
                paths = filtered_paths

            elif "*" not in file_pattern and not self._file_repo.file_exists(file_pattern):
                return SearchCodeResponse(
                    message=f"No file found on path {file_pattern}.",
                    hits=[],
                )
            else:
                return SearchCodeResponse(
                    message=f"No files found for file pattern {file_pattern}.",
                    hits=[],
                )

        logger.info(
            f"find_by_name(class_name={class_name}, function_name={function_name}, file_pattern={file_pattern}) {len(paths)} hits."
        )

        if not paths:
            if function_name:
                return SearchCodeResponse(message=f"No functions found with the name {function_name}.")
            else:
                return SearchCodeResponse(message=f"No classes found with the name {class_name}.")

        filtered_out_by_class_name = 0
        invalid_blocks = 0

        if category and category != "test":
            filtered_paths = []
            for file_path, block_path in paths:
                if not is_test(file_path):
                    filtered_paths.append((file_path, block_path))

            filtered_out_test_files = len(paths) - len(filtered_paths)
            if filtered_out_test_files > 0:
                logger.debug(f"find_by_name() Filtered out {filtered_out_test_files} test files.")

            paths = filtered_paths

        hits = 0
        files_with_spans = {}
        for file_path, block_path in paths:
            file = self._file_repo.get_file(file_path)
            if not file:
                logger.warning(
                    f"find_by_name(function_name: {function_name}, class_name: {class_name}, file_pattern: {file_pattern}) Could not find file {file_path}."
                )
                continue
            elif not file.module:
                logger.warning(
                    f"find_by_name(funtion_name: {function_name}, class_name: {class_name}, file_pattern: {file_pattern}) Could not parse module for file {file_path}."
                )
                continue

            found_block = file.module.find_by_path(block_path)

            if not found_block:
                invalid_blocks += 1
                continue

            blocks = []

            # If a class name was provided we only search for the class
            if class_name and found_block.type == CodeBlockType.CLASS:
                # if function names are provided we filter on the function names
                if function_name:
                    function_block = found_block.find_by_identifier(function_name)
                    if function_block:
                        blocks.append(function_block)
                else:
                    for child in found_block.children:
                        blocks.append(child)
            else:
                blocks.append(found_block)

            if blocks:
                if file_path not in files_with_spans:
                    files_with_spans[file_path] = SearchCodeHit(file_path=file_path)

                for block in blocks:
                    if block.belongs_to_span.span_id in files_with_spans[file_path].span_ids:
                        continue

                    hits += 1

                    files_with_spans[file_path].add_span(
                        block.belongs_to_span.span_id,
                        rank=0,
                        tokens=block.belongs_to_span.tokens,
                    )

        if filtered_out_by_class_name > 0:
            logger.debug(
                f"find_by_function_name() Filtered out {filtered_out_by_class_name} functions by class name {class_name}."
            )

        if invalid_blocks > 0:
            logger.debug(f"find_by_function_name() Ignored {invalid_blocks} invalid blocks.")

        if paths and function_name:
            message = f"Found {len(paths)} functions."
        elif paths and class_name:
            message = f"Found {len(paths)} classes."
        elif class_name and function_name:
            message = f"No function found with the name {function_name} in class {class_name}."
        elif class_name:
            message = f"No classe found with the name {class_name}."
        elif function_name:
            message = f"No function found with the names {function_name}."
        else:
            message = "No results found."

        file_paths = [file.file_path for file in files_with_spans.values()]
        if file_pattern:
            file_paths = await self.matching_files(file_pattern)
            file_paths = _rerank_files(file_paths, file_pattern)

        search_hits = []
        for rank, file_path in enumerate(file_paths):
            file = files_with_spans.get(file_path)
            if file:
                for span in file.spans:
                    span.rank = rank
                search_hits.append(file)

        return SearchCodeResponse(
            message=message,
            hits=search_hits,
        )

    @tracer.start_as_current_span("vector_search")
    async def _vector_search(
        self,
        query: str = "",
        exact_query_match: bool = False,
        category: str | None = None,
        file_pattern: Optional[str] = None,
        exact_content_match: Optional[str] = None,
        top_k: int = 500,
    ):
        # Import llama_index components only when needed
        from llama_index.core.vector_stores.types import VectorStoreQuery

        if file_pattern:
            query += f" file:{file_pattern}"

        if exact_content_match:
            query += "\n" + exact_content_match

        if not query:
            raise ValueError("At least one of query, span_keywords or content_keywords must be provided.")

        logger.debug(f"vector_search() Searching for query [{query[:50]}...] and file pattern [{file_pattern}].")

        # TODO: Make this async
        query_embedding = self._embed_model.get_query_embedding(query)

        # FIXME: Filters can't be used ATM. Category isn't set in some instance vector stores
        # filters = MetadataFilters(filters=[], condition=FilterCondition.AND)
        # if category:
        #    filters.filters.append(MetadataFilter(key="category", value=category))

        query_bundle = VectorStoreQuery(
            query_str=query,
            query_embedding=query_embedding,
            similarity_top_k=top_k,  # TODO: Fix paging?
            #    filters=filters,
        )

        result = self._vector_store.query(query_bundle)

        filtered_out_snippets = 0
        ignored_removed_snippets = 0
        sum_tokens = 0

        sum_tokens_per_file = {}

        if file_pattern:
            include_files = await self.matching_files(file_pattern)
            if len(include_files) == 0:
                logger.info(f"vector_search() No files found for file pattern {file_pattern}, return empty result...")
                return []
        else:
            include_files = []

        search_results = []

        for node_id, distance in zip(result.ids, result.similarities, strict=False):
            node_doc = self._docstore.get_document(node_id, raise_error=False)
            if not node_doc:
                ignored_removed_snippets += 1
                # TODO: Retry to get top_k results
                continue

            is_test_file = is_test(node_doc.metadata["file_path"])
            if category and category != "test" and is_test_file:
                filtered_out_snippets += 1
                continue

            if include_files and node_doc.metadata["file_path"] not in include_files:
                filtered_out_snippets += 1
                continue

            if category == "implementation" and is_test_file:
                filtered_out_snippets += 1
                continue

            if category == "test" and not is_test_file:
                filtered_out_snippets += 1
                continue

            if exact_query_match and query not in node_doc.get_content():
                filtered_out_snippets += 1
                continue

            if exact_content_match and not is_string_in(exact_content_match, node_doc.get_content()):
                filtered_out_snippets += 1
                continue

            if node_doc.metadata["file_path"] not in sum_tokens_per_file:
                sum_tokens_per_file[node_doc.metadata["file_path"]] = 0

            sum_tokens += node_doc.metadata["tokens"]
            sum_tokens_per_file[node_doc.metadata["file_path"]] += node_doc.metadata["tokens"]

            code_snippet = CodeSnippet(
                id=node_doc.id_,
                file_path=node_doc.metadata["file_path"],
                distance=distance,
                content=node_doc.get_content(),
                tokens=node_doc.metadata["tokens"],
                span_ids=node_doc.metadata.get("span_ids", []),
                start_line=node_doc.metadata.get("start_line", None),
                end_line=node_doc.metadata.get("end_line", None),
            )

            search_results.append(code_snippet)

        # TODO: Rerank by file pattern if no exact matches on file pattern

        logger.debug(
            f"vector_search() Returning {len(search_results)} search results. "
            f"(Ignored {ignored_removed_snippets} removed search results. "
            f"Filtered out {filtered_out_snippets} search results from vector search result with {len(result.ids)} hits.)"
        )

        return search_results

    def run_ingestion(
        self,
        repo_path: Optional[str] = None,
        input_files: list[str] | None = None,
        num_workers: Optional[int] = None,
    ):
        # Import llama_index components only when needed

        repo_path = repo_path or self._file_repo.path

        # Only extract file name and type to not trigger unnecessary embedding jobs
        def file_metadata_func(file_path: str) -> dict:
            file_path = file_path.replace(repo_path, "")
            if file_path.startswith("/"):
                file_path = file_path[1:]

            category = "test" if is_test(file_path) else "implementation"

            return {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": mimetypes.guess_type(file_path)[0],
                "category": category,
            }

        if self._settings and self._settings.language == "java":
            required_exts = [".java"]
        else:
            required_exts = [".py"]

        if input_files:
            input_files = [os.path.join(repo_path, file) for file in input_files if not file.startswith(repo_path)]

        try:
            reader = SimpleDirectoryReader(
                input_dir=repo_path,
                file_metadata=file_metadata_func,
                input_files=input_files,
                filename_as_id=True,
                required_exts=required_exts,
                recursive=True,
            )
        except Exception as e:
            logger.exception(
                f"Failed to create reader with input_dir {repo_path}, input_files {input_files} and required_exts {required_exts}."
            )
            raise e

        embed_pipeline = IngestionPipeline(
            transformations=[self._embed_model],
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            docstore=self._docstore,
            vector_store=self._vector_store,
        )

        docs = reader.load_data()
        logger.info(f"Read {len(docs)} documents")

        blocks_by_class_name = {}
        blocks_by_function_name = {}

        def index_callback(codeblock: CodeBlock):
            if codeblock.type == CodeBlockType.CLASS:
                if codeblock.identifier not in blocks_by_class_name:
                    blocks_by_class_name[codeblock.identifier] = []
                blocks_by_class_name[codeblock.identifier].append((codeblock.module.file_path, codeblock.full_path()))

            if codeblock.type == CodeBlockType.FUNCTION:
                if codeblock.identifier not in blocks_by_function_name:
                    blocks_by_function_name[codeblock.identifier] = []
                blocks_by_function_name[codeblock.identifier].append(
                    (codeblock.module.file_path, codeblock.full_path())
                )

        from moatless.index.epic_split import EpicSplitter

        splitter = EpicSplitter(
            language=self._settings.language,
            min_chunk_size=self._settings.min_chunk_size,
            chunk_size=self._settings.chunk_size,
            hard_token_limit=self._settings.hard_token_limit,
            max_chunks=self._settings.max_chunks,
            comment_strategy=self._settings.comment_strategy,
            index_callback=index_callback,
            repo_path=repo_path,
        )

        prepared_nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
        prepared_tokens = sum([count_tokens(node.get_content(), self._settings.embed_model) for node in prepared_nodes])
        logger.info(f"Run embed pipeline with {len(prepared_nodes)} nodes and {prepared_tokens} tokens")

        embedded_nodes = embed_pipeline.run(nodes=list(prepared_nodes), show_progress=True, num_workers=num_workers)
        embedded_tokens = sum([count_tokens(node.get_content(), self._settings.embed_model) for node in embedded_nodes])
        logger.info(f"Embedded {len(embedded_nodes)} vectors with {embedded_tokens} tokens")

        self._code_block_index = CodeBlockIndex(blocks_by_class_name, blocks_by_function_name)

        return len(embedded_nodes), embedded_tokens

    def persist(self, persist_dir: str):
        self._vector_store.persist(persist_dir)
        self._docstore.persist(os.path.join(persist_dir, DEFAULT_PERSIST_FNAME))
        self._settings.persist(persist_dir)
        self._code_block_index.persist(persist_dir)


def _rerank_files(file_paths: list[str], file_pattern: str):
    if len(file_paths) < 2:
        return file_paths

    tokenized_query = file_pattern.replace(".py", "").replace("*", "").split("/")
    tokenized_query = [part for part in tokenized_query if part.strip()]
    query = "/".join(tokenized_query)

    scored_files = []
    for file_path in file_paths:
        cleaned_file_path = file_path.replace(".py", "")
        score = fuzz.partial_ratio(cleaned_file_path, query)
        scored_files.append((file_path, score))

    scored_files.sort(key=lambda x: x[1], reverse=True)

    sorted_file_paths = [file for file, score in scored_files]

    logger.info(
        f"rerank_files() Reranked {len(file_paths)} files with query {tokenized_query}. First hit {sorted_file_paths[0]}"
    )

    return sorted_file_paths


def is_string_in(s1, s2):
    s1_clean = s1.replace(" ", "").replace("\t", "").replace("\n", "")
    s2_clean = s2.replace(" ", "").replace("\t", "").replace("\n", "")
    found_in = s1_clean in s2_clean
    return found_in
