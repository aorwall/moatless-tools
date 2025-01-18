import argparse
import json
import logging
import os
from typing import Optional, Dict, List

from dotenv import load_dotenv
from moatless.benchmark.swebench import (
    get_repo_dir_name,
)
from moatless.benchmark.swebench.utils import create_repository
from moatless.benchmark.utils import get_moatless_instances
from moatless.index.settings import IndexSettings
from moatless.index.code_index import CodeIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_persist_dir(instance_id: str, index_store_dir: str) -> str:
    return os.path.join(index_store_dir, get_repo_dir_name(instance_id))

def load_previous_instances(report_path: str, instance_by_id: Dict) -> Dict:
    previous_instances = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            for line in f:
                report = json.loads(line)
                previous_instance = instance_by_id[report["instance_id"]]
                previous_instances[previous_instance["repo"]] = previous_instance
                del instance_by_id[report["instance_id"]]
    return previous_instances

def get_previous_instance(instance_id: str, instance_by_id: Dict) -> Optional[Dict]:
    """Get the previous instance based on numeric ID ordering within the same repo."""
    current_num = extract_number(instance_id)
    # Get the prefix (everything before the last number)
    prefix = instance_id.rsplit('-', 1)[0]
    
    # Filter instances to only those from same repo (matching prefix)
    prev_instances = [
        inst for inst in instance_by_id.values()
        if inst["instance_id"].startswith(prefix) and 
        extract_number(inst["instance_id"]) < current_num
    ]
    
    if not prev_instances:
        return None
    return max(prev_instances, key=lambda x: extract_number(x["instance_id"]))

def create_index(
    instance: Dict,
    instance_by_id: Dict,
    index_settings: IndexSettings,
    index_store_dir: str
) -> CodeIndex:
    repository = create_repository(instance)
    previous_instance = get_previous_instance(instance["instance_id"], instance_by_id)
    
    if previous_instance:
        logger.info(f"Loading index for {instance['instance_id']} from previous instance {previous_instance['instance_id']}")
        persist_dir = get_persist_dir(previous_instance["instance_id"], index_store_dir)
        
        if not os.path.exists(persist_dir):
            raise RuntimeError(
                f"Previous instance index not found at {persist_dir}. "
                f"Cannot load index for instance {instance['instance_id']}"
            )
            
        return CodeIndex.from_persist_dir(persist_dir, file_repo=repository)
    return CodeIndex(file_repo=repository, settings=index_settings)

def ingest_instance(
    instance: Dict,
    code_index: CodeIndex,
    index_store_dir: str,
    num_workers: int
) -> tuple[int, int]:
    logger.info(f"Processing instance: {instance['instance_id']}")

    vectors, indexed_tokens = code_index.run_ingestion(num_workers=num_workers)
    logger.info(f"Indexed {vectors} vectors and {indexed_tokens} tokens")
    
    persist_dir = get_persist_dir(instance["instance_id"], index_store_dir)
    code_index.persist(persist_dir=persist_dir)
    logger.info(f"Index persisted to {persist_dir}")
    
    return vectors, indexed_tokens

def write_report(
    report_path: str,
    instance: Dict,
    vectors: int,
    indexed_tokens: int
) -> None:
    with open(report_path, "a") as f:
        report = {
            "instance_id": instance["instance_id"],
            "vectors": vectors,
            "indexed_tokens": indexed_tokens,
        }
        f.write(json.dumps(report) + "\n")

def process_instances(
    instances: List[Dict],
    instance_by_id: Dict,
    index_settings: IndexSettings,
    index_store_dir: str,
    report_path: str,
    num_workers: int
) -> None:
    for instance in instances:
        persist_dir = get_persist_dir(instance["instance_id"], index_store_dir)
        
        # Skip if index already exists
        if os.path.exists(persist_dir):
            logger.info(f"Skipping instance {instance['instance_id']} - index already exists at {persist_dir}")
            continue
            
        logger.info(f"Processing instance: {instance['instance_id']}")
        
        code_index = create_index(
            instance,
            instance_by_id,
            index_settings,
            index_store_dir
        )
        
        vectors, indexed_tokens = ingest_instance(
            instance,
            code_index,
            index_store_dir,
            num_workers
        )
        
        write_report(report_path, instance, vectors, indexed_tokens)

def extract_number(instance_id: str) -> int:
    """Extract the numeric portion of an instance ID for sorting."""
    return int(instance_id.split('-')[-1])

def main():
    parser = argparse.ArgumentParser(description="Ingest and create code index for SWE-Bench instances")
    parser.add_argument(
        "--index-store-dir",
        required=True,
        help="Directory to store vector index files"
    )
    parser.add_argument(
        "--embed-model",
        default="voyage-code-2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Specific instance IDs to process"
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset to load instances from"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--report-path",
        default="ingest_report.jsonl",
        help="Path to write ingestion reports"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing"
    )
    parser.add_argument(
        "--prefix",
        help="Process all instances with this prefix"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize index settings
    index_settings = IndexSettings(embed_model=args.embed_model, dimensions=1024)
    
    # Load instances
    instance_by_id = get_moatless_instances()
    
    if args.prefix:
        # Filter and sort instances by prefix
        instances = sorted(
            [inst for inst in instance_by_id.values() 
             if inst["instance_id"].startswith(args.prefix)],
            key=lambda x: extract_number(x["instance_id"])
        )
        logger.info(f"Processing {len(instances)} instances matching prefix {args.prefix}")
    elif args.instance_ids:
        # Sort instance IDs numerically
        sorted_instance_ids = sorted(args.instance_ids, key=extract_number)
        # Filter to only requested instances
        instances = [
            instance_by_id[instance_id] 
            for instance_id in sorted_instance_ids 
            if instance_id in instance_by_id
        ]
        logger.info(f"Processing {len(instances)} specified instances")
    else:
        # Sort all instances by numeric ID
        instances = sorted(
            instance_by_id.values(),
            key=lambda x: extract_number(x["instance_id"])
        )
        logger.info(f"Processing {len(instances)} instances")
    
    # Process all instances
    process_instances(
        instances,
        instance_by_id,
        index_settings,
        args.index_store_dir,
        args.report_path,
        args.num_workers
    )

if __name__ == "__main__":
    main() 