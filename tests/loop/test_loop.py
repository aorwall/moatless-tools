import os
import tempfile

import pytest

from moatless import AgenticLoop
from moatless.benchmark.swebench import create_workspace, load_instance
from moatless.repository import GitRepository
from moatless.settings import Settings
from moatless.trajectory import Trajectory

pytest.mark.api_keys_required = pytest.mark.skipif(
    "VOYAGE_API_KEY" not in os.environ or os.environ["VOYAGE_API_KEY"] == "",
    reason="VOYAGE_API_KEY environment variable is required"
)


@pytest.mark.api_keys_required
def test_rerun_save_and_load_trajectory():
    trajectory = Trajectory.load("tests/trajectories/django__django_16379.json")
    Settings.cheap_model = None  # To not use an LLM when generating commit messages

    # Start by running the trajectory again with mocked action requests
    instance = load_instance("django__django-16379")
    workspace = create_workspace(instance)
    assert isinstance(workspace.file_repo, GitRepository)
    mocked_actions = trajectory.get_mocked_actions()
    expected_states = trajectory.get_expected_states()

    loop = AgenticLoop(
        trajectory.transition_rules, workspace=workspace, mocked_actions=mocked_actions, expected_states=expected_states
    )
    response = loop.run(message=trajectory.initial_message)

    assert workspace.file_context.has_span(
        "django/core/cache/backends/filebased.py", "FileBasedCache.has_key"
    )
    assert loop.workspace.file_repo._initial_commit != loop.workspace.file_repo._current_commit
    diff = loop.workspace.file_repo.diff()
    assert diff == """diff --git a/django/core/cache/backends/filebased.py b/django/core/cache/backends/filebased.py
index 631da49444..f980d8d6ac 100644
--- a/django/core/cache/backends/filebased.py
+++ b/django/core/cache/backends/filebased.py
@@ -91,8 +91,11 @@ class FileBasedCache(BaseCache):
     def has_key(self, key, version=None):
         fname = self._key_to_file(key, version)
         if os.path.exists(fname):
-            with open(fname, "rb") as f:
-                return not self._is_expired(f)
+            try:
+                with open(fname, "rb") as f:
+                    return not self._is_expired(f)
+            except FileNotFoundError:
+                return False
         return False
 
     def _cull(self):"""

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        loop.persist(tmp_file.name)

        # Verify that the loop can be iniatied from the saved trajectory
        saved_loop = AgenticLoop.from_trajectory_file(tmp_file.name)

        saved_response = saved_loop.run(message=trajectory.initial_message)
        assert saved_response == response
        assert saved_loop.workspace.file_repo._initial_commit == loop.workspace.file_repo._initial_commit
        assert saved_loop.workspace.file_repo._current_commit == loop.workspace.file_repo._current_commit
        assert saved_loop.workspace.file_repo.diff() == loop.workspace.file_repo.diff()
