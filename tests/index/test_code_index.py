from moatless.benchmark.swebench import setup_swebench_repo
from moatless.benchmark.utils import get_moatless_instance
from moatless.index import IndexSettings, CodeIndex
from moatless.index.settings import CommentStrategy
from moatless.repository import GitRepository, FileRepository


def test_ingestion():
    index_settings = IndexSettings(
        embed_model="voyage-code-2",
        dimensions=1536,
        language="python",
        min_chunk_size=200,
        chunk_size=750,
        hard_token_limit=3000,
        max_chunks=200,
        comment_strategy=CommentStrategy.ASSOCIATE,
    )

    instance_id = "django__django-12419"
    instance = get_moatless_instance(instance_id, split="verified")
    repo_dir = setup_swebench_repo(instance)
    print(repo_dir)
    repo = FileRepository(repo_dir)
    code_index = CodeIndex(settings=index_settings, file_repo=repo)

    vectors, indexed_tokens = code_index.run_ingestion(num_workers=1, input_files=["django/conf/global_settings.py"])

    results = code_index._vector_search("SECURE_REFERRER_POLICY setting")

    for result in results:
        print(result)
