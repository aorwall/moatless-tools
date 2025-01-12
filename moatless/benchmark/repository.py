import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional, List

from moatless.benchmark.schema import Evaluation, EvaluationInstance, DateTimeEncoder

logger = logging.getLogger(__name__)


class EvaluationRepository(ABC):
    """Abstract base class for evaluation persistence."""

    @abstractmethod
    def save_evaluation(self, evaluation: Evaluation) -> None:
        """Save an evaluation."""
        pass

    @abstractmethod
    def load_evaluation(self, evaluation_name: str) -> Evaluation:
        """Load an evaluation."""
        pass

    @abstractmethod
    def save_instance(self, evaluation_name: str, instance: EvaluationInstance) -> None:
        """Save an instance."""
        pass

    @abstractmethod
    def load_instance(
        self, evaluation_name: str, instance_id: str
    ) -> Optional[EvaluationInstance]:
        """Load an instance."""
        pass

    @abstractmethod
    def delete_instance(self, evaluation_name: str, instance_id: str) -> None:
        """Delete an instance."""
        pass

    @abstractmethod
    def list_instances(self, evaluation_name: str) -> List[EvaluationInstance]:
        """List all instances for an evaluation."""
        pass

    @abstractmethod
    def list_evaluations(self) -> List[str]:
        """List all evaluation names from storage."""
        pass


class EvaluationFileRepository(EvaluationRepository):
    """File-based implementation of evaluation repository."""

    def __init__(self, evaluations_dir: str):
        self.evaluations_dir = evaluations_dir
        self._repo_lock = threading.Lock()  # Add lock for repository operations

    def get_evaluation_dir(self, evaluation_name: str) -> str:
        """Get the directory path for an evaluation."""
        return os.path.join(self.evaluations_dir, evaluation_name)

    def save_evaluation(self, evaluation: Evaluation) -> None:
        """Save an evaluation to disk."""
        eval_dir = self.get_evaluation_dir(evaluation.evaluation_name)
        if not os.path.exists(eval_dir):
            logger.debug(f"Creating evaluation directory: {eval_dir}")
            os.makedirs(eval_dir, exist_ok=True)

        eval_file = os.path.join(eval_dir, "evaluation.json")
        logger.info(f"Saving evaluation {evaluation.evaluation_name} to {eval_file}")
        with open(eval_file, "w") as f:
            json.dump(evaluation.model_dump(), f, cls=DateTimeEncoder, indent=2)

    def load_evaluation(self, evaluation_name: str) -> Evaluation | None:
        """Load an evaluation from disk."""
        eval_path = os.path.join(
            self.get_evaluation_dir(evaluation_name), "evaluation.json"
        )
        logger.debug(f"Attempting to load evaluation from: {eval_path}")
        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation file not found: {eval_path}")
            return None

        try:
            logger.debug(f"Reading evaluation file: {eval_path}")
            with open(eval_path, "r") as f:
                data = json.load(f)
                evaluation = Evaluation.model_validate(data)
                logger.debug(
                    f"Successfully loaded evaluation {evaluation_name} with status {evaluation.status}"
                )
                return evaluation
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in evaluation file {eval_path}: {e}")
            raise ValueError(f"Invalid JSON in evaluation file: {e}")
        except Exception as e:
            logger.error(f"Error loading evaluation from {eval_path}: {e}")
            raise ValueError(f"Error loading evaluation: {e}")

    def save_instance(self, evaluation_name: str, instance: EvaluationInstance) -> None:
        """Save an instance to disk."""
        with self._repo_lock:
            evaluation = self.load_evaluation(evaluation_name)
            # TODO: Add or update instance on evaluation

    def load_instance(
        self, evaluation_name: str, instance_id: str
    ) -> Optional[EvaluationInstance]:
        """Load an instance from disk."""

        evaluation = self.load_evaluation(evaluation_name)
        return next(
            instance
            for instance in evaluation.instances
            if instance.instance_id == instance_id
        )

    def delete_instance(self, evaluation_name: str, instance_id: str) -> None:
        """Delete an instance directory."""

        # TODO: Update evaluation.instances

    def list_instances(self, evaluation_name: str) -> List[EvaluationInstance]:
        """List all instances for an evaluation."""
        evaluation = self.load_evaluation(evaluation_name)
        return evaluation.instances

    def list_evaluations(self) -> List[str]:
        """List all evaluation names from disk."""
        if not os.path.exists(self.evaluations_dir):
            logger.debug(
                f"Evaluations directory does not exist: {self.evaluations_dir}"
            )
            return []

        # Use os.listdir to get all directories and filter for those that have evaluation.json
        all_dirs = os.listdir(self.evaluations_dir)
        logger.debug(f"All directories in {self.evaluations_dir}: {all_dirs}")

        eval_dirs = [
            d
            for d in all_dirs
            if os.path.isdir(os.path.join(self.evaluations_dir, d))
            and os.path.exists(os.path.join(self.evaluations_dir, d, "evaluation.json"))
        ]
        logger.debug(f"Found evaluation directories: {eval_dirs}")
        return sorted(eval_dirs)  # Sort for consistent ordering
