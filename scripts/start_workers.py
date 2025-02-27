#!/usr/bin/env python3

import sys
import subprocess
import signal
import time
import os
import psutil
from typing import List
import redis
from rq import Worker
from rq.worker import WorkerStatus

def cleanup_existing_workers(name_prefix: str) -> None:
    """Clean up any existing worker processes and their Redis registrations."""
    # First clean up processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            # Look for rq worker processes with matching name prefix
            if cmdline and 'rq' in cmdline and 'worker' in cmdline:
                for arg in cmdline:
                    if arg.startswith('-n') or arg.startswith('--name'):
                        worker_name_idx = cmdline.index(arg) + 1
                        if worker_name_idx < len(cmdline):
                            worker_name = cmdline[worker_name_idx]
                            if worker_name.startswith(name_prefix):
                                print(f"Killing existing worker process {proc.info['pid']} ({worker_name})")
                                try:
                                    os.killpg(os.getpgid(proc.info['pid']), signal.SIGKILL)
                                except (ProcessLookupError, psutil.NoSuchProcess):
                                    pass
                                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Then clean up Redis registrations
    try:
        print("Cleaning up Redis registrations")
        redis_conn = redis.Redis()
        workers = Worker.all(connection=redis_conn)
        for worker in workers:
            print(f"Worker {worker.name} is in state {worker.state}")
            if worker.name.startswith(name_prefix):
                print(f"Cleaning up Redis registration for worker {worker.name}")
                try:
                    # Clean up worker from Redis
                    worker.register_death()
                except Exception as e:
                    print(f"Error cleaning up Redis for worker {worker.name}: {e}")
    except redis.ConnectionError:
        print("Warning: Could not connect to Redis to clean up worker registrations")

    # Give Redis a moment to process the cleanup
    time.sleep(1)

def get_active_worker_names() -> set:
    """Get set of active worker names from Redis."""
    try:
        redis_conn = redis.Redis()
        workers = Worker.all(connection=redis_conn)
        return {w.name for w in workers if w.state not in (WorkerStatus.DEAD, WorkerStatus.CRASHED)}
    except redis.ConnectionError:
        print("Warning: Could not connect to Redis to check existing workers")
        return set()

def is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def start_workers(num_workers: int, name_prefix: str = "worker") -> List[subprocess.Popen]:
    # Clean up any existing workers first
    cleanup_existing_workers(name_prefix)
    
    # Brief pause to allow processes to be cleaned up
    time.sleep(1)
    
    workers = []
    for i in range(num_workers):
        worker_name = f"{name_prefix}_{i+1}"
        print(f"Starting worker {worker_name}")
        try:
            worker = subprocess.Popen(
                ["rq", "worker", "-n", worker_name],
                preexec_fn=os.setsid  # Create new process group
            )
            print(f"Worker {worker_name} started with PID {worker.pid}")
            workers.append(worker)
            # Brief pause to allow worker to start
            time.sleep(0.5)
        except Exception as e:
            print(f"Error starting worker {worker_name}: {str(e)}")
            continue
            
    if not workers:
        print("No workers could be started successfully")
        sys.exit(1)
        
    return workers

def terminate_workers(workers: List[subprocess.Popen], timeout: int = 5) -> None:
    """Terminate workers with a timeout, using SIGKILL if needed."""
    print("\nTerminating workers...")
    
    try:
        # First attempt graceful shutdown with SIGTERM
        for worker in workers:
            try:
                if worker.poll() is None and is_process_running(worker.pid):
                    print(f"Sending SIGTERM to process group of worker {worker.pid}")
                    os.killpg(os.getpgid(worker.pid), signal.SIGTERM)
                else:
                    print(f"Worker {worker.pid} is already terminated")
            except (ProcessLookupError, psutil.NoSuchProcess):
                print(f"Worker {worker.pid} no longer exists")
                continue
    
        # Wait for workers to terminate
        termination_start = time.time()
        while time.time() - termination_start < timeout:
            if all(not is_process_running(w.pid) for w in workers):
                print("All workers terminated successfully")
                break
            time.sleep(0.1)
    
        # Force kill any remaining workers
        remaining_workers = [w for w in workers if is_process_running(w.pid)]
        if remaining_workers:
            print(f"\nFound {len(remaining_workers)} workers still running after SIGTERM")
            for worker in remaining_workers:
                try:
                    if is_process_running(worker.pid):
                        print(f"Force killing worker {worker.pid} with SIGKILL")
                        os.killpg(os.getpgid(worker.pid), signal.SIGKILL)
                except (ProcessLookupError, psutil.NoSuchProcess):
                    print(f"Worker {worker.pid} terminated before SIGKILL")
                    continue

        # Final wait to ensure all processes are cleaned up
        for worker in workers:
            try:
                if worker.poll() is None:
                    worker.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if is_process_running(worker.pid):
                    print(f"Warning: Worker {worker.pid} may still be running")
                else:
                    print(f"Worker {worker.pid} is confirmed terminated")
    
    except KeyboardInterrupt:
        # If interrupted during cleanup, force kill everything immediately
        print("\nInterrupted during cleanup, force killing all remaining workers...")
        for worker in workers:
            try:
                if is_process_running(worker.pid):
                    print(f"Force killing worker {worker.pid}")
                    os.killpg(os.getpgid(worker.pid), signal.SIGKILL)
                    worker.wait(timeout=1)
            except (ProcessLookupError, psutil.NoSuchProcess, subprocess.TimeoutExpired):
                continue

def signal_handler(signum, frame):
    # The global workers variable will be set in main
    global workers
    terminate_workers(workers)
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python start_workers.py <number_of_workers> [name_prefix]")
        sys.exit(1)

    try:
        num_workers = int(sys.argv[1])
        if num_workers < 1:
            raise ValueError("Number of workers must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    name_prefix = sys.argv[2] if len(sys.argv) == 3 else "worker"

    # Initialize global workers variable
    workers = []
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting {num_workers} workers with name prefix '{name_prefix}'...")
    workers = start_workers(num_workers, name_prefix)
    
    # Keep the script running and wait for all workers
    try:
        for worker in workers:
            worker.wait()
    except KeyboardInterrupt:
        terminate_workers(workers) 