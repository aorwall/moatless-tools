#!/usr/bin/env python3
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import cast
from datasets import Dataset, load_dataset

from swebench.harness.constants import SWEbenchInstance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dataset(name: str, split: str) -> list[SWEbenchInstance]:
    dataset = cast(Dataset, load_dataset(name, split=split))
    return [SWEbenchInstance(**instance) for instance in dataset]


def load_swebench_datasets() -> list[SWEbenchInstance]:
    instances = []
    instances.extend(download_dataset("princeton-nlp/SWE-bench_Lite", "test"))
    instances.extend(download_dataset("princeton-nlp/SWE-bench_Verified", "test"))
    instances.extend(download_dataset("SWE-Gym/SWE-Gym", "train"))
    return instances


def create_dataset() -> bool:
    instances_dir = Path("instances")

    try:
        logger.info("Loading SWE-bench dataset...")
        instances = load_swebench_datasets()

        instances_dir.mkdir(exist_ok=True)

        logger.info("Saving individual instance files...")
        for instance in instances:
            instance_path = instances_dir / f"{instance['instance_id']}.json"
            with instance_path.open("w", encoding="utf-8") as f:
                json.dump(instance, f, indent=2)

        logger.info(f"Successfully saved {len(instances)} instances")
        return True

    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        return False


def main():
    success = create_dataset()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
