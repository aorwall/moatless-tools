import logging
import shutil
import tempfile

from moatless.repository import FileRepository
from moatless.benchmark.swebench import load_instances, setup_swebench_repo
import json

from moatless.benchmark.utils import calculate_estimated_context_window
from moatless.index.settings import IndexSettings, CommentStrategy
from moatless.index.code_index import CodeIndex
from dotenv import load_dotenv
from moatless.benchmark.swebench import get_repo_dir_name
import os


index_store_dir = "/home/albert/.moatless/index_stores/20240814-voyage-code-2"

logger = logging.getLogger(__name__)
evaluation_report = "report.jsonl"


def create_instance_list():
    # lite_instance_by_id = load_instances("princeton-nlp/SWE-bench_Lite", split="test")
    instance_by_id = load_instances("princeton-nlp/SWE-bench_Verified", split="test")

    logger.info(
        f"Number of instances: {len(instance_by_id)} from {len(instance_by_id)} SWE-bench_Lite and SWE-bench_Verified"
    )

    instances = list(instance_by_id.values())
    # instances = [instance for instance in instances if instance["instance_id"] in white_list]
    instances = sorted(instances, key=lambda x: x["created_at"])

    logger.info(f"Number of instances: {len(instances)}")
    return instances


# with open("index_eval.csv", "w") as f:
#    f.write("instance_id,vectors,indexed_tokens,all_matching_context_window,any_matching_context_window\n")


previous_instances = {}


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

load_dotenv("../.env")


def get_persist_dir(instance):
    return os.path.join(index_store_dir, get_repo_dir_name(instance["instance_id"]))


def ingest(code_index, instance):
    vectors, indexed_tokens = code_index.run_ingestion(num_workers=1)
    logger.info(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")

    persist_dir = get_persist_dir(instance)
    code_index.persist(persist_dir=persist_dir)
    logger.info(f"Index persisted to {persist_dir}")
    return vectors, indexed_tokens


def evaluate_index(code_index, instance):
    results = code_index._vector_search(instance["problem_statement"], top_k=1000)

    expected_changes, sum_tokens = calculate_estimated_context_window(instance, results)
    all_matching_context_window = None
    any_matching_context_window = None

    expected_matches = [
        context for context in expected_changes if context["context_window"] is not None
    ]
    if expected_matches:
        all_matching_context_window = max(
            context["context_window"] for context in expected_matches
        )
        any_matching_context_window = min(
            context["context_window"] for context in expected_matches
        )

        if len(expected_matches) == len(expected_changes):
            logger.info(
                f"Found all expected changes within a context window of {all_matching_context_window} tokens, first match at context window {any_matching_context_window}"
            )
        else:
            any_matching_context_window = min(
                context["context_window"]
                for context in expected_changes
                if context["context_window"] is not None
            )
            logger.info(
                f"Found {len(expected_matches)} expected changes within a context window {all_matching_context_window} tokens, first match at context window {any_matching_context_window} max context window {sum_tokens} tokens"
            )
    else:
        logger.info(
            f"No expected changes found in context window of {sum_tokens} tokens"
        )

    for change in expected_changes:
        if change["context_window"] is None:
            logger.info(
                f"Expected change: {change['file_path']} ({change['start_line']}-{change['end_line']}) not fund, closest match: {change.get('closest_match_lines')}"
            )
        else:
            logger.info(
                f"Expected change: {change['file_path']} ({change['start_line']}-{change['end_line']}) found at context window {change['context_window']} tokens. Distance: {change['distance']}. Position: {change['position']}"
            )

    return expected_changes, all_matching_context_window, any_matching_context_window


def write_report(
    instance,
    expected_changes,
    vectors,
    indexed_tokens,
    all_matching_context_window,
    any_matching_context_window,
):
    with open("report.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance["instance_id"],
                    "vectors": vectors,
                    "indexed_tokens": indexed_tokens,
                    "all_matching_context_window": all_matching_context_window,
                    "any_matching_context_window": any_matching_context_window,
                    "expected_changes": expected_changes,
                }
            )
            + "\n"
        )

    with open("index_eval_2.csv", "a") as f:
        f.write(
            f"{instance['instance_id']},{vectors},{indexed_tokens},{all_matching_context_window},{any_matching_context_window}\n"
        )


def run_indexing():
    # lite_instance_by_id = load_instances("princeton-nlp/SWE-bench_Lite", split="test")
    instance_by_id = load_instances("princeton-nlp/SWE-bench_Verified", split="test")

    # instance_by_id = {**lite_instance_by_id, **verified_instance_by_id}
    # logger.info(
    #    f"Number of instances: {len(instance_by_id)} from {len(lite_instance_by_id)} SWE-bench_Lite and {len(verified_instance_by_id)} SWE-bench_Verified")

    instances = list(instance_by_id.values())
    instances = sorted(instances, key=lambda x: x["created_at"])

    logger.info(f"Number of instances: {len(instances)}")

    if os.path.exists(evaluation_report):
        with open(evaluation_report, "r") as f:
            for line in f:
                report = json.loads(line)
                previous_instance = instance_by_id.get(report["instance_id"])
                if previous_instance:
                    previous_instances[previous_instance["repo"]] = previous_instance
                    del instance_by_id[report["instance_id"]]

    for i, instance in enumerate(instances):
        logger.info(
            f"Processing instance {i + 1}/{len(instances)}: {instance['instance_id']} {instance['created_at']}"
        )

        persist_dir = get_persist_dir(instance)

        code_index = None

        if os.path.exists(persist_dir):
            logger.info(f"Index exists on {persist_dir}")
            # try:
            #    logger.info(f"Loading index from {persist_dir}")
            #    code_index = CodeIndex.from_persist_dir(persist_dir, file_repo=repo)
            # except Exception as e:
            #    logger.error(f"Error loading index: {e}")
        else:
            logger.info(f"No index found at {persist_dir}")

            repo_dir = setup_swebench_repo(instance)
            repo = FileRepository(repo_dir)
            # if not code_index:
            previous_instance = previous_instances.get(instance["repo"])
            if previous_instance:
                logger.info(
                    f"Loading previous index from {get_persist_dir(previous_instance)}"
                )
                code_index = CodeIndex.from_persist_dir(
                    get_persist_dir(previous_instance), file_repo=repo
                )
            else:
                code_index = CodeIndex(settings=index_settings, file_repo=repo)

            vectors, indexed_tokens = ingest(code_index, instance)
            (
                expected_changes,
                all_matching_context_window,
                any_matching_context_window,
            ) = evaluate_index(code_index, instance)
            write_report(
                instance,
                expected_changes,
                vectors,
                indexed_tokens,
                all_matching_context_window,
                any_matching_context_window,
            )

        previous_instances[instance["repo"]] = instance

if "main" == __name__:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_indexing()
