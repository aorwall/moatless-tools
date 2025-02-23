import asyncio
import os
import sys
import time
import threading
from typing import Optional
import logging
logger = logging.getLogger(__name__)

class BlockingDetector:
    def __init__(self, threshold_ms: float = 1000):
        self.threshold_sec = threshold_ms / 1000
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check = 0
        self._startup_complete = False
        self._blocking_stats = {}  # Track blocking statistics
        
    async def start(self):
        if self._task is not None:
            return
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
        return self._blocking_stats

    def _update_stats(self, operation_type: str, filename: str, function_name: str, elapsed: float):
        """Update blocking statistics."""
        key = f"{operation_type}:{os.path.basename(filename)}:{function_name}"
        if key not in self._blocking_stats:
            self._blocking_stats[key] = {
                'count': 0,
                'total_time': 0.0,
                'max_time': 0.0,
                'last_seen': None
            }
        
        stats = self._blocking_stats[key]
        stats['count'] += 1
        stats['total_time'] += elapsed
        stats['max_time'] = max(stats['max_time'], elapsed)
        stats['last_seen'] = time.time()

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
                    thread = threading.get_ident() if thread_id == threading.get_ident() else f"Thread-{thread_id}"
                    
                    stack = []
                    current = frame
                    skip_until_non_fs = False
                    last_fs_frame = None
                    fs_context_frames = []
                    llm_operation = None
                    llm_context_frames = []
                    
                    while current:
                        filename = current.f_code.co_filename
                        name = current.f_code.co_name
                        
                        # Skip monitoring code and known system files that aren't interesting
                        if any(x in filename for x in (
                            'runner.py', 'run_api.py', 'moatless/api/__init__.py',
                            'asyncio', 'threading.py', 'queue.py', 
                            'uvicorn/server.py', '_monitor',
                            'concurrent/futures',
                            'logging/__init__.py', 'argparse.py'
                        )):
                            current = current.f_back
                            continue
                            
                        # Skip uvicorn startup
                        if 'uvicorn' in filename and name in ('run', 'serve'):
                            if not self._startup_complete and 'server.py' in filename:
                                self._startup_complete = True
                            current = current.f_back
                            continue
                            
                        # Handle filesystem operations specially
                        is_fs_op = ('pathlib.py' in filename and 
                                  any(op in name for op in ('glob', 'scandir', '_iterate_directories', '_select_from')))
                        
                        if is_fs_op:
                            if not last_fs_frame:
                                last_fs_frame = current
                                # Update stats for filesystem operation
                                self._update_stats('fs', filename, name, elapsed)
                            skip_until_non_fs = True
                            fs_context_frames.append(current)
                            current = current.f_back
                            continue
                        elif last_fs_frame:
                            # Keep collecting frames after filesystem operation until we hit system files
                            if not any(x in filename for x in ('site-packages', 'dist-packages', 'lib/python')):
                                fs_context_frames.append(current)
                            
                        # Handle LLM operations
                        is_llm_related = ('litellm' in filename or 
                                        any(x in filename for x in ('openai', 'anthropic', 'completion')))
                        
                        if is_llm_related:
                            if not llm_operation and name == 'completion':
                                llm_operation = current
                                # Update stats for LLM operation
                                self._update_stats('llm', filename, name, elapsed)
                            llm_context_frames.append(current)
                        elif llm_operation:
                            # Keep collecting frames after LLM operation until we hit system files
                            if not any(x in filename for x in ('site-packages', 'dist-packages', 'lib/python')):
                                llm_context_frames.append(current)
                            
                        if skip_until_non_fs:
                            skip_until_non_fs = False
                            # Add a summary of the filesystem operation with context
                            if last_fs_frame:
                                fs_name = last_fs_frame.f_code.co_name
                                fs_locals = last_fs_frame.f_locals
                                summary_parts = [f"  [File operation: {fs_name}"]
                                
                                if 'pattern' in fs_locals:
                                    summary_parts.append(f" pattern={fs_locals['pattern']}")
                                if 'parent_path' in fs_locals:
                                    summary_parts.append(f" in {fs_locals['parent_path']}")
                                    
                                # Add call context
                                app_frames = [f for f in fs_context_frames 
                                           if not any(x in f.f_code.co_filename 
                                                    for x in ('site-packages', 'dist-packages', 'lib/python', 'pathlib.py'))]
                                if app_frames:
                                    last_app_frame = app_frames[-1]
                                    summary_parts.append(f" called from {os.path.basename(last_app_frame.f_code.co_filename)}:{last_app_frame.f_code.co_name}")
                                    
                                    # Show the actual application frames that led to this
                                    stack.append("".join(summary_parts) + "]")
                                    for app_frame in reversed(app_frames):
                                        stack.append(f"  File \"{app_frame.f_code.co_filename}\", line {app_frame.f_lineno}, in {app_frame.f_code.co_name}")
                                        
                                        # Show relevant locals for the app frames
                                        interesting_locals = {}
                                        for k, v in app_frame.f_locals.items():
                                            if k.startswith('__') or callable(v):
                                                continue
                                            try:
                                                str_val = str(v)
                                                if len(str_val) > 100:
                                                    str_val = str_val[:97] + "..."
                                                interesting_locals[k] = str_val
                                            except:
                                                continue
                                        
                                        if interesting_locals:
                                            stack.append("  Locals:")
                                            for k, v in list(interesting_locals.items())[:5]:
                                                stack.append(f"    {k} = {v}")
                                else:
                                    stack.append("".join(summary_parts) + "]")
                                
                                last_fs_frame = None
                            
                        frame_info = f"  File \"{filename}\", line {current.f_lineno}, in {name}"
                        stack.append(frame_info)
                            
                        # Show relevant locals for debugging
                        if elapsed > 1.0:
                            interesting_locals = {}
                            for k, v in current.f_locals.items():
                                # Skip magic methods, callables, and known noisy objects
                                if k.startswith('__') or callable(v) or k in ('config', 'log_config'):
                                    continue
                                    
                                # For LLM operations and context, be more verbose
                                if current in llm_context_frames:
                                    # Always include these for LLM context
                                    if k in ('model', 'messages', 'temperature', 'max_tokens', 'timeout',
                                           'prompt', 'query', 'text', 'content'):
                                        try:
                                            if k == 'messages':
                                                msg_count = len(v)
                                                total_length = sum(len(str(m.get('content', ''))) for m in v)
                                                interesting_locals['message_stats'] = f"{msg_count} messages, {total_length} chars total"
                                                # Also show first message content truncated
                                                if msg_count > 0:
                                                    first_content = str(v[0].get('content', ''))[:200]
                                                    if len(first_content) > 197:
                                                        first_content = first_content[:197] + "..."
                                                    interesting_locals['first_message'] = first_content
                                            else:
                                                interesting_locals[k] = v
                                        except:
                                            continue
                                    continue
                                    
                                # Convert value to string and truncate if too long
                                try:
                                    str_val = str(v)
                                    if len(str_val) > 100:  # Truncate long values
                                        str_val = str_val[:97] + "..."
                                    interesting_locals[k] = str_val
                                except:
                                    continue  # Skip values that can't be converted to string
                                    
                            if interesting_locals:
                                stack.append("  Locals:")
                                # Only show up to 5 most relevant locals
                                for k, v in list(interesting_locals.items())[:5]:
                                    stack.append(f"    {k} = {v}")
                                    
                        current = current.f_back
                        
                    if stack:
                        # If this was an LLM operation, add a summary at the top
                        if llm_operation:
                            locals_dict = llm_operation.f_locals
                            summary_parts = ["  [LLM Operation:"]
                            if 'model' in locals_dict:
                                summary_parts.append(f" model={locals_dict['model']}")
                            if 'timeout' in locals_dict:
                                summary_parts.append(f" timeout={locals_dict['timeout']}s")
                            if 'temperature' in locals_dict:
                                summary_parts.append(f" temp={locals_dict['temperature']}")
                            
                            # Add call context
                            if llm_context_frames:
                                app_frames = [f for f in llm_context_frames 
                                           if not any(x in f.f_code.co_filename 
                                                    for x in ('site-packages', 'dist-packages', 'lib/python'))]
                                if app_frames:
                                    last_app_frame = app_frames[-1]
                                    summary_parts.append(f" called from {os.path.basename(last_app_frame.f_code.co_filename)}:{last_app_frame.f_code.co_name}")
                            
                            summary_parts.append("]")
                            stack.insert(0, "".join(summary_parts))
                            
                        blocking_info.append(
                            f"=== {thread} ===\n"
                            f"Stack (most recent call last):\n" + 
                            "\n".join(reversed(stack))
                        )
                
                if blocking_info:
                    logger.warning(
                        f"Event loop blocked for {elapsed*1000:.1f}ms\n" +
                        '\n'.join(blocking_info)
                    )
            
            self._last_check = now
