#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
from pathlib import Path

from moatless.workspace import Workspace

# Add moatless-tools to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from moatless.actions.read_files import ReadFiles, ReadFilesArgs
from moatless.environment.local import LocalBashEnvironment
from moatless.file_context import FileContext
from moatless.repository.file import FileRepository
from moatless.utils.tokenizer import count_tokens

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # Set up the target repository path
    repo_path = "/Users/albert/repos/fp/tm-api"
    target_directory = "src/*/java/se/frankpenny/tm/core"

    # Create the environment
    env = LocalBashEnvironment(cwd=repo_path)

    # Create the file repository
    repository = FileRepository(repo_path=repo_path)

    # Create file context
    file_context = FileContext(repo=repository)

    # Set up the ReadFiles action
    read_files_action = ReadFiles()

    # Create a mock workspace with the environment
    workspace = Workspace(repository=repository, environment=env)
    read_files_action._workspace = workspace

    # Create the arguments for reading all Java files in the target directory
    # Set max_files=0 to read all files and max_lines_per_file=0 to read all lines
    args = ReadFilesArgs(
        glob_pattern=f"{target_directory}/**/*.java",
        max_files=0,  # Read all files
        max_lines_per_file=0,  # Read all lines
        thoughts="Reading all Java files in the core directory",
    )

    # Execute the action
    logger.info(f"Reading Java files in {target_directory}...")
    result = await read_files_action.execute(args, file_context)

    # Check if we got a result
    if result is None or result.message is None:
        logger.error("No results returned from read_files_action")
        return

    # Print result message
    print(result.message)

    # Count tokens in the result
    token_count = count_tokens(result.message)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Token count: {token_count}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
