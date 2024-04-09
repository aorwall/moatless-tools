import logging
import os
import uuid
from typing import List

from llama_index.embeddings.voyageai import VoyageEmbedding
from pydantic import BaseModel

from moatless.coder import Coder
from moatless.ingestion import CodeBaseIngestionPipeline
from moatless.planner import Planner
from moatless.select_blocks import CodeBlockSelector
from moatless.splitters.epic_split import EpicSplitter
from moatless.types import ContextFile, BaseResponse, DevelopmentTask


class PipelineStep(BaseModel):
    action: str
    response: dict


class PipelineState(BaseModel):
    coding_request: str = None
    context_files: List[ContextFile] = []
    tasks: List[DevelopmentTask] = []
    steps: List[PipelineStep] = []


logger = logging.getLogger(__name__)


class CodingPipeline:

    def __init__(self,
                 path: str,
                 coding_request: str,
                 context_files: List[ContextFile],
                 pipeline_dir: str = None):
        self._path = path
        self._pipeline_dir = pipeline_dir

        if len(context_files) > 1:
            raise ValueError("Only one context file is supported at the moment.")

        self._state = PipelineState(context_files=context_files, coding_request=coding_request)

        if self._pipeline_dir:
            self._state_file_path = f"{self._pipeline_dir}/state.json"
            if os.path.exists(self._state_file_path):
                with open(self._state_file_path, 'r') as f:
                    self._state = PipelineState.model_validate_json(f.read())

        voyage_api_key = os.environ.get("VOYAGE_API_KEY", "your-api-key")
        voyage_embedding = VoyageEmbedding(
            model_name="voyage-code-2", voyage_api_key=voyage_api_key, truncation=False
        )

        self.ingestion_pipeline = CodeBaseIngestionPipeline.from_path(
            path=self._path,
            splitter=EpicSplitter(chunk_size=1000, min_chunk_size=100, language="python"),
            embed_model=voyage_embedding
        )

        self._block_selector = CodeBlockSelector(
            repo_path=self._path,
            model_name='openrouter/anthropic/claude-3-haiku')
# openrouter/anthropic/claude-3-haiku
# claude-3-haiku-20240307
        self._planner = Planner(
            repo_path=self._path,
            model_name='gpt-4-0125-preview')

        self._coder = Coder(
            repo_path=self._path,
            model_name='gpt-4-0125-preview')

    def run(self):
        logger.info("Running pipeline...")

        if not self._state.context_files:
            logger.info("No context files found, trigger find_files agent")
            logger.warning("...which is not implemented yet.")
            # TODO: Implement find_files agent
            raise NotImplementedError("find_files agent is not implemented yet. ")

        self.select_blocks()
        self.plan()
        self.write_code()

    def select_blocks(self):
        logger.info("Selecting blocks...")
        for context_file in self._state.context_files:
            if context_file.block_paths is not None:
                logger.info(f"Block paths already selected in file {context_file.file_path}.")
                return False

            response = self._block_selector.select_blocks(
                instructions=self._state.coding_request,
                file_path=context_file.file_path)

            context_file.block_paths = response.block_paths

            self._state.steps.append(PipelineStep(action='select_blocks', response=response.dict()))
            self._persist()

        return True

    def plan(self):
        logger.info("Planning development...")
        if not self._state.context_files:
            raise ValueError("No context files found. Cannot plan development.")

        planned_tasks = [task for task in self._state.tasks if task.state == 'planned']
        if planned_tasks:
            logger.info("Tasks already planned.")
            return False

        response = self._planner.plan_development(
            instructions=self._state.coding_request,
            files=self._state.context_files)

        self._state.tasks.extend(response.tasks)
        self._state.steps.append(PipelineStep(action='plan', response=response.dict()))
        self._persist()

        return True

    def write_code(self):
        logger.info("Writing code...")

        planned_tasks = [task for task in self._state.tasks if task.state == 'planned']
        if not planned_tasks:
            logger.info("Won't run code step as there are no planned tasks to implement.")
            return False

        for task in planned_tasks:
            if task.action != 'update':
                logger.info(f"Skipping task {task.action} in {task.file_path} as it's not an update action.")
                task.state = 'rejected'
                continue

            logger.info(f"Updating {task.file_path}::{'.'.join(task.block_path)}...")
            response = self._coder.write_code(
                instructions=task.instructions,
                file_path=task.file_path,
                block_path=task.block_path)

            if response.error:
                task.state = 'failed'
            elif not response.diff:
                task.state = 'rejected'
            else:
                task.state = 'completed'

            self._state.steps.append(PipelineStep(action='write_code', response=response.dict()))
            self._persist()

        return True

    def _persist(self):
        if self._pipeline_dir:
            state_path = f"{self._pipeline_dir}/state.json"
            with open(state_path, 'w') as f:
                f.write(self._state.model_dump_json(indent=2))
