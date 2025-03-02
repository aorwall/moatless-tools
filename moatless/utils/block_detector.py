import asyncio
import logging
import os
import re
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BlockingDetector:
    def __init__(self, threshold_ms: float = 1000):
        self.threshold_sec = threshold_ms / 1000
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check = 0
        self._startup_complete = False
        self._blocking_stats = {}  # Track blocking statistics
        self._fs_patterns = defaultdict(int)  # Track filesystem access patterns
        self._fs_hot_paths = defaultdict(float)  # Track paths with most blocking time
        self._runner_thread_id = None  # Store the runner's thread ID
        self._runner_task_id = None  # Store the runner's task ID

    async def start(self):
        if self._task is not None:
            return

        # Identify the runner's context
        current_frame = sys._getframe()
        while current_frame:
            if "runner.py" in current_frame.f_code.co_filename:
                self._runner_thread_id = threading.get_ident()
                current_task = asyncio.current_task()
                if current_task:
                    self._runner_task_id = id(current_task)
                break
            current_frame = current_frame.f_back

        if not self._runner_thread_id:
            logger.warning("BlockingDetector started outside of runner.py context")

        self.running = True
        self._task = asyncio.create_task(self._monitor())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def get_blocking_statistics(self) -> dict:
        """Return statistics about blocking operations."""
        stats = {
            "operations": self._blocking_stats,
            "fs_patterns": dict(self._fs_patterns),
            "hot_paths": dict(
                sorted(self._fs_hot_paths.items(), key=lambda x: x[1], reverse=True)[:10]
            ),  # Top 10 hottest paths
        }
        return stats

    def _update_stats(
        self, operation_type: str, filename: str, function_name: str, elapsed: float, extra_data: dict = None
    ):
        """Update blocking statistics with enhanced tracking."""
        key = f"{operation_type}:{os.path.basename(filename)}:{function_name}"
        if key not in self._blocking_stats:
            self._blocking_stats[key] = {
                "count": 0,
                "total_time": 0.0,
                "max_time": 0.0,
                "last_seen": None,
                "patterns": defaultdict(int),
                "slow_operations": [],  # Track particularly slow operations
            }

        stats = self._blocking_stats[key]
        stats["count"] += 1
        stats["total_time"] += elapsed
        stats["max_time"] = max(stats["max_time"], elapsed)
        stats["last_seen"] = time.time()

        if extra_data:
            # Track filesystem patterns
            if "path" in extra_data:
                path = extra_data["path"]
                pattern = self._extract_path_pattern(path)
                stats["patterns"][pattern] += 1
                self._fs_patterns[pattern] += 1

                # Update hot paths tracking
                self._fs_hot_paths[path] += elapsed

            # Track slow operations
            if elapsed > self.threshold_sec:
                slow_op = {"timestamp": time.time(), "elapsed": elapsed, "details": extra_data}
                stats["slow_operations"].append(slow_op)
                # Keep only last 10 slow operations
                stats["slow_operations"] = stats["slow_operations"][-10:]

    def _extract_path_pattern(self, path: str) -> str:
        """Extract a pattern from a file path to identify common access patterns."""
        # Replace numbers with #
        pattern = re.sub(r"\d+", "#", path)
        # Replace UUIDs with @UUID@
        pattern = re.sub(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "@UUID@", pattern, flags=re.IGNORECASE
        )
        # Replace hash-like strings with @HASH@
        pattern = re.sub(r"[a-f0-9]{32,}", "@HASH@", pattern, flags=re.IGNORECASE)
        return pattern

    def _get_frame_context(self, frame) -> dict:
        """Extract relevant context from a frame."""
        context = {
            "filename": frame.f_code.co_filename,
            "function": frame.f_code.co_name,
            "lineno": frame.f_lineno,
            "locals": {},
        }

        # Extract relevant locals
        for k, v in frame.f_locals.items():
            if k.startswith("__"):
                continue
            try:
                if isinstance(v, (str, int, float, bool)):
                    context["locals"][k] = v
                elif hasattr(v, "name"):  # Often useful for file objects
                    context["locals"][k] = f"<{v.__class__.__name__}:{v.name}>"
                # Special handling for LLM context
                elif k in (
                    "model",
                    "messages",
                    "temperature",
                    "max_tokens",
                    "timeout",
                    "prompt",
                    "query",
                    "text",
                    "content",
                ):
                    if k == "messages" and isinstance(v, (list, tuple)):
                        msg_count = len(v)
                        total_length = sum(len(str(m.get("content", ""))) for m in v)
                        context["locals"]["message_stats"] = f"{msg_count} messages, {total_length} chars total"
                        if msg_count > 0:
                            first_content = str(v[0].get("content", ""))[:200]
                            if len(first_content) > 197:
                                first_content = first_content[:197] + "..."
                            context["locals"]["first_message"] = first_content
                    else:
                        context["locals"][k] = str(v)[:100]
            except:
                continue

        return context

    def _is_relevant_frame(self, frame, thread_id: int) -> bool:
        """Determine if a frame is relevant for monitoring."""
        # If we have a runner thread ID, only monitor that thread
        if self._runner_thread_id and thread_id != self._runner_thread_id:
            return False

        filename = frame.f_code.co_filename
        name = frame.f_code.co_name

        # Skip internal/system frames
        if any(
            x in filename
            for x in (
                "asyncio",
                "threading.py",
                "queue.py",
                "opentelemetry",
                "concurrent/futures",
                "logging/__init__.py",
                "argparse.py",
            )
        ):
            return False

        # Skip monitoring code itself
        if "_monitor" in name or "block_detector.py" in filename:
            return False

        # Skip uvicorn startup
        if "uvicorn" in filename and name in ("run", "serve"):
            if not self._startup_complete and "server.py" in filename:
                self._startup_complete = True
            return False

        return True

    async def _monitor(self):
        self._last_check = time.monotonic()

        while self.running:
            await asyncio.sleep(0.01)

            now = time.monotonic()
            elapsed = now - self._last_check

            if elapsed > self.threshold_sec:
                frames = sys._current_frames()

                blocking_info = []
                for thread_id, frame in frames.items():
                    # Skip if not relevant thread
                    if not self._is_relevant_frame(frame, thread_id):
                        continue

                    thread = threading.get_ident() if thread_id == threading.get_ident() else f"Thread-{thread_id}"

                    stack = []
                    current = frame
                    fs_operations = []  # Track filesystem operations in this stack
                    llm_operations = []  # Track LLM operations in this stack

                    while current:
                        if not self._is_relevant_frame(current, thread_id):
                            current = current.f_back
                            continue

                        filename = current.f_code.co_filename
                        name = current.f_code.co_name

                        # Enhanced filesystem operation detection
                        is_fs_op = (
                            "pathlib.py" in filename
                            and any(
                                op in name
                                for op in ("glob", "scandir", "_iterate_directories", "_select_from", "exists", "stat")
                            )
                        ) or ("os.py" in filename and any(op in name for op in ("stat", "listdir", "walk", "scandir")))

                        # LLM operation detection
                        is_llm_op = "litellm" in filename or any(
                            x in filename for x in ("openai", "anthropic", "completion")
                        )

                        context = self._get_frame_context(current)

                        if is_fs_op:
                            # Extract path information
                            path = None
                            if "self" in current.f_locals and hasattr(current.f_locals["self"], "__str__"):
                                path = str(current.f_locals["self"])
                            elif "path" in current.f_locals:
                                path = str(current.f_locals["path"])

                            if path:
                                context["path"] = path
                                self._update_stats("fs", filename, name, elapsed, {"path": path, "context": context})

                            fs_operations.append(context)

                        if is_llm_op:
                            if name == "completion":
                                self._update_stats("llm", filename, name, elapsed, {"context": context})
                            llm_operations.append(context)

                        frame_info = f'  File "{filename}", line {current.f_lineno}, in {name}'
                        stack.append(frame_info)

                        # Show relevant locals for debugging
                        if elapsed > self.threshold_sec:
                            if context["locals"]:
                                stack.append("  Locals:")
                                for k, v in list(context["locals"].items())[:5]:
                                    stack.append(f"    {k} = {v}")

                        current = current.f_back

                    if stack:
                        # Add operation summaries
                        summaries = []

                        if fs_operations:
                            summary = ["  [Filesystem Operations:"]
                            for op in fs_operations:
                                if "path" in op:
                                    summary.append(f"\n    - {op['function']} on {op['path']}")
                            summary.append("  ]")
                            summaries.append("\n".join(summary))

                        if llm_operations:
                            summary = ["  [LLM Operations:"]
                            for op in llm_operations:
                                summary_parts = [f"\n    - {op['function']}"]
                                if "model" in op["locals"]:
                                    summary_parts.append(f" model={op['locals']['model']}")
                                if "message_stats" in op["locals"]:
                                    summary_parts.append(f" ({op['locals']['message_stats']})")
                                summary.append("".join(summary_parts))
                            summary.append("  ]")
                            summaries.append("\n".join(summary))

                        if summaries:
                            stack.insert(0, "\n".join(summaries))

                        blocking_info.append(
                            f"=== {thread} ===\n" f"Stack (most recent call last):\n" + "\n".join(reversed(stack))
                        )

                if blocking_info:
                    # Enhanced logging with pattern analysis
                    patterns = dict(self._fs_patterns)
                    hot_paths = dict(
                        sorted(self._fs_hot_paths.items(), key=lambda x: x[1], reverse=True)[:5]
                    )  # Top 5 hottest paths

                    logger.warning(
                        f"Event loop blocked for {elapsed*1000:.1f}ms\n"
                        + "Common filesystem patterns:\n"
                        + "\n".join(f"  - {pattern}: {count} times" for pattern, count in patterns.items())
                        + "\n\nHottest paths:\n"
                        + "\n".join(f"  - {path}: {time:.1f}ms total" for path, time in hot_paths.items())
                        + "\n\nStack traces:\n"
                        + "\n".join(blocking_info)
                    )

            self._last_check = now
