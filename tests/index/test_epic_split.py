import cProfile
import pstats

from llama_index.core import SimpleDirectoryReader

from moatless.benchmark.swebench import setup_swebench_repo
from moatless.benchmark.utils import get_moatless_instance
from moatless.index import IndexSettings
from moatless.index.epic_split import EpicSplitter


def test_epic_split():
    instance_id = "django__django-16139"
    instance = get_moatless_instance(instance_id)
    repo_path = setup_swebench_repo(instance)

    file = "tests/admin_views/tests.py"
    input_files = [f"{repo_path}/{file}"]

    settings = IndexSettings()
    splitter = EpicSplitter(
        language=settings.language,
        min_chunk_size=settings.min_chunk_size,
        chunk_size=settings.chunk_size,
        hard_token_limit=settings.hard_token_limit,
        max_chunks=settings.max_chunks,
        comment_strategy=settings.comment_strategy,
        min_lines_to_parse_block=50,
        repo_path=repo_path,
    )

    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        input_files=input_files,
        filename_as_id=True,
        recursive=True,
    )
    docs = reader.load_data()
    print(f"Read {len(docs)} documents")

    # Profile the get_nodes_from_documents method
    profiler = cProfile.Profile()
    profiler.enable()
    prepared_nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(40)

    for node in prepared_nodes:
        print(
            f"{node.id_} {node.metadata['tokens']} {node.metadata['start_line']}-{node.metadata['end_line']} {node.metadata['span_ids']}"
        )


def test_impl_spans():
    instance_id = "django__django-12419"
    instance = get_moatless_instance(instance_id, split="verified")
    repo_path = setup_swebench_repo(instance)

    file = "django/conf/global_settings.py"
    input_files = [f"{repo_path}/{file}"]

    settings = IndexSettings()
    splitter = EpicSplitter(
        language=settings.language,
        min_chunk_size=settings.min_chunk_size,
        chunk_size=settings.chunk_size,
        hard_token_limit=settings.hard_token_limit,
        max_chunks=settings.max_chunks,
        comment_strategy=settings.comment_strategy,
        min_lines_to_parse_block=50,
        repo_path=repo_path,
    )

    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        input_files=input_files,
        filename_as_id=True,
        recursive=True,
    )
    docs = reader.load_data()
    print(f"Read {len(docs)} documents")

    # Profile the get_nodes_from_documents method
    profiler = cProfile.Profile()
    profiler.enable()
    prepared_nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(40)

    for node in prepared_nodes:
        print(
            f"{node.id_} {node.metadata['tokens']} {node.metadata['start_line']}-{node.metadata['end_line']} {node.metadata['span_ids']}"
        )
