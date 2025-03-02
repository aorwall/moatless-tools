import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from moatless.benchmark.swebench.utils import repository_exists
from moatless.evaluation.schema import EvaluationInstance, InstanceStatus
from moatless.evaluation.utils import get_moatless_instance

logger = logging.getLogger(__name__)


class InvalidStateTransition(Exception):
    pass


class StateManager:
    """
    Handles instance state transitions and repository-setup tracking in one place.
    """

    # Allowed transitions in a single mapping
    ALLOWED_TRANSITIONS = {
        InstanceStatus.PENDING: {InstanceStatus.SETTING_UP, InstanceStatus.RUNNING},
        InstanceStatus.SETTING_UP: {InstanceStatus.PENDING, InstanceStatus.ERROR},
        InstanceStatus.RUNNING: {InstanceStatus.COMPLETED, InstanceStatus.ERROR},
        InstanceStatus.COMPLETED: {InstanceStatus.EVALUATING},
        InstanceStatus.EVALUATING: {InstanceStatus.EVALUATED, InstanceStatus.ERROR},
        # EVALUATED, ERROR are effectively terminal
    }

    def __init__(self):
        # Tracks which repos are currently being set up: repo_name -> set of instance_ids
        self._active_repos: dict[str, set[str]] = {}

    async def check_repository_exists(self, moatless_instance: dict, repo_base_dir: str) -> bool:
        return repository_exists(moatless_instance, repo_base_dir)

    async def can_setup_repo(self, instance_id: str) -> bool:
        moatless_instance = get_moatless_instance(instance_id=instance_id)
        repo = moatless_instance["repo"]
        return repo not in self._active_repos or not self._active_repos[repo]

    async def register_repo_setup(self, instance_id: str) -> None:
        moatless_instance = get_moatless_instance(instance_id=instance_id)
        repo = moatless_instance["repo"]
        if repo not in self._active_repos:
            self._active_repos[repo] = set()
        self._active_repos[repo].add(instance_id)
        logger.info(f"Registered repo setup for instance {instance_id} on repo {repo}")

    async def unregister_repo_setup(self, instance_id: str) -> None:
        moatless_instance = get_moatless_instance(instance_id=instance_id)
        repo = moatless_instance["repo"]
        if repo in self._active_repos:
            self._active_repos[repo].discard(instance_id)
            if not self._active_repos[repo]:
                del self._active_repos[repo]
        logger.info(f"Unregistered repo setup for instance {instance_id} on repo {repo}")

    async def set_status(
        self, instance: EvaluationInstance, new_status: InstanceStatus, error: Optional[str] = None
    ) -> None:
        """
        Central method to handle instance state transitions, with built-in validation.
        Raises an exception if the transition is invalid.
        """
        old_status = instance.status
        allowed = self.ALLOWED_TRANSITIONS.get(old_status, set())
        if new_status not in allowed and old_status not in (InstanceStatus.EVALUATED, InstanceStatus.ERROR):
            # EVALUATED / ERROR are effectively terminal, so we rarely move from there
            raise InvalidStateTransition(f"Cannot transition {old_status} -> {new_status}. Allowed: {allowed}")

        instance.status = new_status

        if new_status == InstanceStatus.ERROR:
            instance.error_at = datetime.now(timezone.utc)
            instance.error = error
        elif new_status == InstanceStatus.EVALUATED:
            instance.evaluated_at = datetime.now(timezone.utc)
        elif new_status == InstanceStatus.RUNNING:
            instance.started_at = datetime.now(timezone.utc)
        elif new_status == InstanceStatus.COMPLETED:
            instance.completed_at = datetime.now(timezone.utc)

        logger.info(f"Instance {instance.instance_id} transitioned: {old_status} -> {new_status}")
