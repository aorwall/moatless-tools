"""
Job diagnostic utilities for troubleshooting Kubernetes job issues.
"""

import logging
import os

from kubernetes import client, config

logger = logging.getLogger(__name__)


def diagnose_job_failure(project_id: str, trajectory_id: str, namespace: str = "moatless-tools"):
    """
    Diagnose why a job failed and collect detailed information.

    Args:
        project_id: The project ID
        trajectory_id: The trajectory ID
        namespace: Kubernetes namespace

    Returns:
        Dictionary with diagnostic information
    """
    try:
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        # Initialize Kubernetes API clients
        batch_v1 = client.BatchV1Api()
        core_v1 = client.CoreV1Api()

        # Generate job ID from project and trajectory IDs
        job_id = f"run-{project_id}-{trajectory_id}"
        job_id = "".join(c if c.isalnum() or c == "-" else "-" for c in job_id.lower())
        if len(job_id) > 63:
            prefix = job_id[:31]
            suffix = job_id[-31:]
            job_id = f"{prefix}-{suffix}"

        # Get job information
        job_info = {}
        try:
            job = batch_v1.read_namespaced_job(name=job_id, namespace=namespace)
            job_info = {
                "name": job.metadata.name,
                "created_at": job.metadata.creation_timestamp,
                "status": {
                    "active": job.status.active if job.status and hasattr(job.status, "active") else 0,
                    "succeeded": job.status.succeeded if job.status and hasattr(job.status, "succeeded") else 0,
                    "failed": job.status.failed if job.status and hasattr(job.status, "failed") else 0,
                    "start_time": job.status.start_time if job.status and hasattr(job.status, "start_time") else None,
                    "completion_time": job.status.completion_time
                    if job.status and hasattr(job.status, "completion_time")
                    else None,
                    "conditions": job.status.conditions if job.status and hasattr(job.status, "conditions") else [],
                },
                "spec": {
                    "backoff_limit": job.spec.backoff_limit,
                    "active_deadline_seconds": job.spec.active_deadline_seconds,
                    "ttl_seconds_after_finished": job.spec.ttl_seconds_after_finished,
                },
            }
        except client.ApiException as e:
            job_info = {"error": f"Failed to retrieve job: {e}"}

        # Get all pods for this job
        pods_info = []
        try:
            pods = core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={job_id}")

            for pod in pods.items:
                pod_info = {
                    "name": pod.metadata.name,
                    "created_at": pod.metadata.creation_timestamp,
                    "node": pod.spec.node_name if pod.spec and pod.spec.node_name else None,
                    "phase": pod.status.phase if pod.status else None,
                    "reason": pod.status.reason if pod.status and pod.status.reason else None,
                    "message": pod.status.message if pod.status and pod.status.message else None,
                    "conditions": [],
                    "container_statuses": [],
                    "events": [],
                }

                # Add pod conditions
                if pod.status and pod.status.conditions:
                    for condition in pod.status.conditions:
                        pod_info["conditions"].append(
                            {
                                "type": condition.type,
                                "status": condition.status,
                                "reason": condition.reason,
                                "message": condition.message,
                            }
                        )

                # Add container statuses
                if pod.status and pod.status.container_statuses:
                    for status in pod.status.container_statuses:
                        container_status = {
                            "name": status.name,
                            "ready": status.ready,
                            "restart_count": status.restart_count,
                            "state": {},
                        }

                        # Add state information
                        if status.state:
                            if status.state.running:
                                container_status["state"] = {
                                    "type": "running",
                                    "started_at": status.state.running.started_at,
                                }
                            elif status.state.waiting:
                                container_status["state"] = {
                                    "type": "waiting",
                                    "reason": status.state.waiting.reason,
                                    "message": status.state.waiting.message,
                                }
                            elif status.state.terminated:
                                container_status["state"] = {
                                    "type": "terminated",
                                    "reason": status.state.terminated.reason,
                                    "message": status.state.terminated.message,
                                    "exit_code": status.state.terminated.exit_code,
                                    "started_at": status.state.terminated.started_at,
                                    "finished_at": status.state.terminated.finished_at,
                                }

                        pod_info["container_statuses"].append(container_status)

                # Get pod events
                try:
                    events = core_v1.list_namespaced_event(
                        namespace=namespace, field_selector=f"involvedObject.name={pod.metadata.name}"
                    )

                    for event in events.items:
                        pod_info["events"].append(
                            {
                                "type": event.type,
                                "reason": event.reason,
                                "message": event.message,
                                "count": event.count,
                                "first_timestamp": event.first_timestamp,
                                "last_timestamp": event.last_timestamp,
                            }
                        )
                except Exception as e:
                    pod_info["events_error"] = str(e)

                # Get pod logs
                try:
                    logs = core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=namespace, container="worker", tail_lines=100
                    )
                    pod_info["logs"] = logs
                except Exception as e:
                    pod_info["logs_error"] = str(e)

                pods_info.append(pod_info)
        except client.ApiException as e:
            pods_info = [{"error": f"Failed to retrieve pods: {e}"}]

        # Combine all diagnostic information
        diagnostics = {"project_id": project_id, "trajectory_id": trajectory_id, "job": job_info, "pods": pods_info}

        # Save diagnostics to file
        output_dir = os.path.join(os.environ.get("MOATLESS_DIR", "."), "diagnostics")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{job_id}-diagnostics.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"JOB DIAGNOSTICS for {job_id}\n")
            f.write("=" * 80 + "\n\n")

            f.write("JOB INFORMATION:\n")
            f.write("-" * 80 + "\n")
            for key, value in job_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("PODS INFORMATION:\n")
            f.write("-" * 80 + "\n")
            for i, pod in enumerate(pods_info):
                f.write(f"Pod #{i+1}: {pod.get('name', 'unknown')}\n")
                f.write("." * 40 + "\n")

                if "error" in pod:
                    f.write(f"Error: {pod['error']}\n")
                    continue

                f.write(f"Phase: {pod.get('phase')}\n")
                f.write(f"Node: {pod.get('node')}\n")
                f.write(f"Created: {pod.get('created_at')}\n")

                if pod.get("reason"):
                    f.write(f"Reason: {pod.get('reason')}\n")

                if pod.get("message"):
                    f.write(f"Message: {pod.get('message')}\n")

                f.write("\nConditions:\n")
                for condition in pod.get("conditions", []):
                    f.write(f"  {condition.get('type')}: {condition.get('status')}")
                    if condition.get("reason"):
                        f.write(f" - {condition.get('reason')}")
                    if condition.get("message"):
                        f.write(f" ({condition.get('message')})")
                    f.write("\n")

                f.write("\nContainer Statuses:\n")
                for container in pod.get("container_statuses", []):
                    f.write(
                        f"  {container.get('name')} (ready: {container.get('ready')}, restarts: {container.get('restart_count')})\n"
                    )

                    state = container.get("state", {})
                    if state:
                        f.write(f"    State: {state.get('type', 'unknown')}\n")
                        if state.get("reason"):
                            f.write(f"    Reason: {state.get('reason')}\n")
                        if state.get("message"):
                            f.write(f"    Message: {state.get('message')}\n")
                        if state.get("exit_code") is not None:
                            f.write(f"    Exit Code: {state.get('exit_code')}\n")

                f.write("\nEvents:\n")
                for event in pod.get("events", []):
                    f.write(
                        f"  {event.get('last_timestamp')} [{event.get('type')}] {event.get('reason')}: {event.get('message')} (x{event.get('count', 1)})\n"
                    )

                f.write("\nLogs:\n")
                if "logs" in pod:
                    f.write("```\n")
                    f.write(pod.get("logs", "No logs available"))
                    f.write("\n```\n")
                elif "logs_error" in pod:
                    f.write(f"Error retrieving logs: {pod.get('logs_error')}\n")
                else:
                    f.write("No logs available\n")

                f.write("\n")

        logger.info(f"Diagnostics saved to {filepath}")
        return {"diagnostics_file": filepath, **diagnostics}

    except Exception as e:
        logger.exception(f"Error diagnosing job failure: {e}")
        return {"error": str(e)}


def create_priority_class():
    """
    Create a high-priority PriorityClass if it doesn't exist.
    """
    try:
        # Load Kubernetes configuration
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        # Initialize Kubernetes API client
        api = client.SchedulingV1Api()

        # Check if high-priority PriorityClass exists
        try:
            api.read_priority_class(name="high-priority")
            logger.info("PriorityClass 'high-priority' already exists")
            return True
        except client.ApiException as e:
            if e.status != 404:
                raise

            # Create PriorityClass if it doesn't exist
            priority_class = client.V1PriorityClass(
                api_version="scheduling.k8s.io/v1",
                kind="PriorityClass",
                metadata=client.V1ObjectMeta(name="high-priority"),
                value=1000000,
                global_default=False,
                description="This priority class is used for critical jobs that should not be evicted.",
            )

            api.create_priority_class(body=priority_class)
            logger.info("Created PriorityClass 'high-priority'")
            return True

    except Exception as e:
        logger.exception(f"Error creating priority class: {e}")
        return False
