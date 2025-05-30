import argparse
import cProfile
import fnmatch
import json
import logging
import mimetypes
import os
import pstats
import difflib
import traceback

from llama_index.core import SimpleDirectoryReader
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import VectorStoreQuery

from moatless.benchmark.swebench import setup_swebench_repo, get_repo_dir_name
from moatless.benchmark.swebench.utils import create_repository
from moatless.evaluation.utils import calculate_estimated_context_window, get_moatless_instance, get_moatless_instances
from moatless.index import IndexSettings, CodeIndex
from moatless.index.simple_faiss import SimpleFaissVectorStore
from moatless.index.epic_split import EpicSplitter

# Add args as a module-level variable
args = None

def get_persist_dir(instance):
    return os.path.join(args.vector_store_dir, get_repo_dir_name(instance["instance_id"]))


def create_index(instance_id: str, index_settings: IndexSettings, num_workers: int):
    instance = get_moatless_instance(instance_id)
    repository = create_repository(instance)

    code_index = CodeIndex(file_repo=repository, settings=index_settings)

    vectors, indexed_tokens = code_index.run_ingestion(num_workers=num_workers)
    print(f"Indexed {vectors} vectors and {indexed_tokens} tokens.")

    persist_dir = get_persist_dir(instance)
    code_index.persist(persist_dir=persist_dir)


def evaluate_index(code_index: CodeIndex, instance: dict):
    query = instance["problem_statement"]
    results = code_index._vector_search(query, top_k=1000)

    expected_changes, sum_tokens = calculate_estimated_context_window(instance, results)
    all_matching_context_window = None
    any_matching_context_window = None

    expected_matches = [context for context in expected_changes if context["context_window"] is not None]
    if expected_matches:
        all_matching_context_window = max(context["context_window"] for context in expected_matches)
        any_matching_context_window = min(context["context_window"] for context in expected_matches)

        if len(expected_matches) == len(expected_changes):
            print(
                f"Found all expected changes within a context window of {all_matching_context_window} tokens, first match at context window {any_matching_context_window}")
        else:
            any_matching_context_window = min(
                context["context_window"] for context in expected_changes if context["context_window"] is not None)
            print(
                f"Found {len(expected_matches)} expected changes within a context window {all_matching_context_window} tokens, first match at context window {any_matching_context_window} max context window {sum_tokens} tokens")


    else:
        print(f"No expected changes found in context window of {sum_tokens} tokens")

    for change in expected_changes:
        if change["context_window"] is None:
            print(
                f"Expected change: {change['file_path']} ({change['start_line']}-{change['end_line']}) not fund, closest match: {change.get('closest_match_lines')}")
        else:
            print(
                f"Expected change: {change['file_path']} ({change['start_line']}-{change['end_line']}) found at context window {change['context_window']} tokens. Distance: {change['distance']}. Position: {change['position']}")

    return expected_changes, all_matching_context_window, any_matching_context_window


def evaluate_instance(instance_id: str) -> dict:
    instance = get_moatless_instance(instance_id)
    
    code_index = CodeIndex.from_persist_dir(get_persist_dir(instance))

    expected_changes, all_matching_context_window, any_matching_context_window = evaluate_index(code_index, instance)
    print(f"All matching context window: {all_matching_context_window}")
    print(f"Any matching context window: {any_matching_context_window}")

    return {
        "instance_id": instance_id,
        "resolved_by": len(instance["resolved_by"]),
        "all_matching_context_window": all_matching_context_window,
        "any_matching_context_window": any_matching_context_window,
    }

def evaluate_instances(instance_ids: list[str]) -> list[dict]:
    # Get the last directory name from vector store path
    store_dir_name = os.path.basename(os.path.normpath(args.vector_store_dir))
    output_file = f"index_eval_{store_dir_name}.csv"
    
    with open(output_file, "w") as f:
        f.write("instance_id,resolved_by,all_matching_context_window,any_matching_context_window\n")

    results = []
    for instance_id in instance_ids:
        try:
            result = evaluate_instance(instance_id)
        except Exception as e:
            print(f"Error evaluating instance {instance_id}: {e}")
            # print the traceback
            import traceback
            print(traceback.format_exc())
            continue

        results.append(result)
        with open(output_file, "a") as f:
            f.write(f"{instance_id},{result['resolved_by']},{result['all_matching_context_window']},{result['any_matching_context_window']}\n")
    return results


def split_and_store(instance_id):
    instance = get_moatless_instance(instance_id, split="verified")
    repo_path = setup_swebench_repo(instance)

    def file_metadata_func(file_path: str) -> dict:
        file_path = file_path.replace(repo_path, "")
        if file_path.startswith("/"):
            file_path = file_path[1:]

        test_patterns = [
            "**/test/**",
            "**/tests/**",
            "**/test_*.py",
            "**/*_test.py",
        ]
        category = (
            "test"
            if any(fnmatch.fnmatch(file_path, pattern) for pattern in test_patterns)
            else "implementation"
        )

        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": mimetypes.guess_type(file_path)[0],
            "category": category,
        }

    required_exts = [".py"]

    settings = IndexSettings()
    splitter = EpicSplitter(
        language=settings.language,
        min_chunk_size=settings.min_chunk_size,
        chunk_size=settings.chunk_size,
        hard_token_limit=settings.hard_token_limit,
        max_chunks=settings.max_chunks,
        comment_strategy=settings.comment_strategy,
        repo_path=repo_path,
    )

    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        file_metadata=file_metadata_func,
        filename_as_id=True,
        required_exts=required_exts,
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

    docstore = SimpleDocumentStore()
    docstore.add_documents(prepared_nodes, store_text=True)

    docstore.persist(os.path.join(args.vector_store_dir, get_repo_dir_name(instance_id)))


def read_store():

    instance_0 = "django__django-16041" # 2022-09-09T10:07:29Z
    instance_1 = "django__django-12325" # 2022-09-10T13:27:38Z
    instance_2 = "django__django-12406" # 2022-09-30T08:51:16Z

    split_and_store(instance_1)
    split_and_store(instance_2)
    docstore1 = SimpleDocumentStore.from_persist_path(os.path.join(args.vector_store_dir, get_repo_dir_name(instance_1)))
    previous_doc_hashes = {doc_id: docstore1.get_document_hash(doc_id) for doc_id in
                          docstore1._kvstore.get_all(collection=docstore1._metadata_collection)}

    docstore2 = SimpleDocumentStore.from_persist_path(os.path.join(args.vector_store_dir, get_repo_dir_name(instance_2)))
    current_doc_hashes = {doc_id: docstore2.get_document_hash(doc_id) for doc_id in docstore2._kvstore.get_all(collection=docstore2._metadata_collection)}

    print(f"  Total documents: {len(current_doc_hashes)}")

    new_doc_ids = set(current_doc_hashes.keys()) - set(previous_doc_hashes.keys())
    removed_doc_ids = set(previous_doc_hashes.keys()) - set(current_doc_hashes.keys())
    changed_doc_ids = {doc_id for doc_id in current_doc_hashes.keys()
                       if doc_id in previous_doc_hashes and current_doc_hashes[doc_id] != previous_doc_hashes[doc_id]}

    print(f"  Changes from previous instance:")
    print(f"    New documents: {len(new_doc_ids)}")
    print(f"    Removed documents: {len(removed_doc_ids)}")
    print(f"    Changed documents: {len(changed_doc_ids)}")

    print("\n  First five changed documents:")
    for doc_id in list(changed_doc_ids):
        print(f"    Document ID: {doc_id}")

        if "global_settings" not in doc_id:
            continue
        print(f"      Old hash: {previous_doc_hashes[doc_id]}")
        print(f"      New hash: {current_doc_hashes[doc_id]}")
        
        old_doc = docstore1.get_document(doc_id)
        new_doc = docstore2.get_document(doc_id)
        
        # Compare content
        content_diff = difflib.unified_diff(
            old_doc.text.splitlines(keepends=True),
            new_doc.text.splitlines(keepends=True),
            fromfile='old_content',
            tofile='new_content',
            n=3  # Context lines
        )

        print("      Content Diff:")
        print(''.join(content_diff))

        # Compare metadata
        print("      Metadata Diff:")
        old_metadata = old_doc.metadata
        new_metadata = new_doc.metadata

        all_keys = set(old_metadata.keys()) | set(new_metadata.keys())
        for key in all_keys:
            old_value = old_metadata.get(key)
            new_value = new_metadata.get(key)
            if old_value != new_value:
                print(f"        {key}:")
                print(f"          Old: {json.dumps(old_value, indent=2)}")
                print(f"          New: {json.dumps(new_value, indent=2)}")

        print()

    print()

    vector_store = SimpleFaissVectorStore.from_persist_dir(os.path.join(args.vector_store_dir, get_repo_dir_name(instance_1)))
    query_bundle = VectorStoreQuery(
        query_str="SECURE_REFERRER_POLICY setting",
        similarity_top_k=100,
    )

    result = vector_store.query(query_bundle)
    for res in result.ids:
        print(res)
    previous_doc_hashes = current_doc_hashes

def load_dataset_instances(dataset_name: str) -> set[str]:
    """Load instance IDs from a dataset file."""
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
    dataset_path = os.path.join(datasets_dir, f"{dataset_name}_dataset.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_path} not found")
    
    with open(dataset_path) as f:
        dataset = json.load(f)
        return set(dataset["instance_ids"])

def extract_number(instance_id: str) -> int:
    """Extract the numeric portion of an instance ID for sorting."""
    return int(instance_id.split('-')[-1])

def main():
    parser = argparse.ArgumentParser(description="Evaluate code index performance")
    parser.add_argument("--vector-store-dir", required=True, help="Directory to store vector index files")
    parser.add_argument("--embed-model", default="voyage-code-3", help="Embedding model to use")
    parser.add_argument("--instance-ids", nargs="*", help="Specific instance IDs to evaluate")
    parser.add_argument("--prefix", help="Process all instances with this prefix")
    parser.add_argument("--dataset", help="Dataset name to load instance IDs from")
    parser.add_argument("--mode", choices=["create", "evaluate", "split", "read"], required=True, 
                       help="Operation mode: create index, evaluate index, split documents, or read store")
    parser.add_argument("--output", default="index_eval.csv", help="Output CSV file for evaluation results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
    
    # Update the global args variable
    global args
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Set up index settings
    index_settings = IndexSettings(
        embed_model=args.embed_model,
        dimensions=1024,
    )
    
    # Get instance IDs either from direct input, prefix, or dataset
    instance_ids = []
    if args.prefix:
        # Get all instances from dataset
        all_instances = get_moatless_instances()
        # Filter by prefix and sort
        matching_instances = sorted(
            [inst for inst in all_instances.values() 
             if inst["instance_id"].startswith(args.prefix)],
            key=lambda x: extract_number(x["instance_id"])
        )
        instance_ids = [inst["instance_id"] for inst in matching_instances]
        print(f"Found {len(instance_ids)} instances matching prefix {args.prefix}")
    elif args.instance_ids:
        instance_ids = args.instance_ids
    elif args.dataset:
        try:
            instance_ids = list(load_dataset_instances(args.dataset))
            print(f"Loaded {len(instance_ids)} instances from dataset {args.dataset}")
        except FileNotFoundError as e:
            print(str(e))
            return
        
    
    if args.mode == "create":
        if not instance_ids:
            print("Error: Must provide instance IDs or dataset for create mode")
            return
        print(f"Will create index for {len(instance_ids)} instances")
        for instance_id in instance_ids:
            create_index(instance_id, index_settings, args.num_workers)
    
    elif args.mode == "evaluate":
        if not instance_ids:
            print("Error: Must provide instance IDs or dataset for evaluate mode")
            return
        results = evaluate_instances(instance_ids)
        
        # Get the last directory name from vector store path for output file
        store_dir_name = os.path.basename(os.path.normpath(args.vector_store_dir))
        output_file = f"index_eval_{store_dir_name}.csv"
        
        # Write results to output file with store dir name
        with open(output_file, "w") as f:
            f.write("instance_id,resolved_by,all_matching_context_window,any_matching_context_window\n")
            for result in results:
                f.write(f"{result['instance_id']},{result['resolved_by']},{result['all_matching_context_window']},{result['any_matching_context_window']}\n")
    
    elif args.mode == "split":
        if not instance_ids:
            print("Error: Must provide instance IDs or dataset for split mode")
            return
        for instance_id in instance_ids:
            try:
                split_and_store(instance_id)
            except Exception as e:
                print(f"Failed to split and store {instance_id}: {e}")
    
    elif args.mode == "read":
        read_store()

if __name__ == "__main__":
    main()