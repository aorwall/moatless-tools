"""API endpoints for runner management."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from moatless.api.dependencies import get_runner
from moatless.api.swebench.schema import RunnerResponseDTO, RunnerStatsDTO
from moatless.runner.runner import BaseRunner, JobsStatusSummary, JobStatus, JobDetails

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("", response_model=RunnerResponseDTO)
async def get_runner_info(
    runner: BaseRunner = Depends(get_runner),
) -> RunnerResponseDTO:
    """Get the runner"""
    return RunnerResponseDTO(info=await runner.get_runner_info(), jobs=await runner.get_jobs())


@router.get("/stats", response_model=RunnerStatsDTO)
async def get_runner_stats(
    runner: BaseRunner = Depends(get_runner),
) -> RunnerStatsDTO:
    """Get lightweight runner stats for the status bar"""
    runner_info = await runner.get_runner_info()
    jobs = await runner.get_jobs()

    # Count jobs by status
    pending_jobs = sum(1 for job in jobs if job.status == JobStatus.PENDING)
    running_jobs = sum(1 for job in jobs if job.status == JobStatus.RUNNING)

    # Get active workers from runner info
    active_workers = runner_info.data.get("ready_nodes", 0)
    total_workers = runner_info.data.get("nodes", 0)

    # Get queue size if the runner supports it
    queue_size = 0
    if hasattr(runner, "get_queue_size"):
        queue_size = await runner.get_queue_size()

    return RunnerStatsDTO(
        runner_type=runner_info.runner_type,
        status=runner_info.status,
        active_workers=active_workers,
        total_workers=total_workers,
        pending_jobs=pending_jobs,
        running_jobs=running_jobs,
        total_jobs=len(jobs),
        queue_size=queue_size,
    )


@router.get("/jobs/summary/{project_id}", response_model=JobsStatusSummary)
async def get_job_status_summary(project_id: str, runner: BaseRunner = Depends(get_runner)) -> JobsStatusSummary:
    """Get a summary of job statuses for a project"""
    return await runner.get_job_status_summary(project_id)


@router.post("/jobs/{project_id}/cancel")
async def cancel_jobs(project_id: str, request: Request, runner: BaseRunner = Depends(get_runner)):
    """Cancel jobs for a project"""
    data = (
        await request.json()
        if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0
        else None
    )
    trajectory_id = data.get("trajectory_id") if data else None
    await runner.cancel_job(project_id, trajectory_id)
    return {"status": "success", "message": "Job(s) canceled successfully"}


@router.post("/jobs/reset")
async def reset_jobs(request: Request, runner: BaseRunner = Depends(get_runner)):
    """Reset all jobs or jobs for a specific project.

    This endpoint will:
    1. Cancel all running jobs
    2. Clear all job history

    Accepts an optional project_id in the request body to reset jobs only for that project.
    """
    data = (
        await request.json()
        if request.headers.get("content-length") and int(request.headers.get("content-length", "0")) > 0
        else None
    )
    project_id = data.get("project_id") if data else None
    success = await runner.reset_jobs(project_id)

    if success:
        return {
            "status": "success",
            "message": f"Jobs reset successfully{f' for project {project_id}' if project_id else ''}",
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to reset jobs",
        )


@router.get("/jobs/{project_id}/{trajectory_id}/details")
async def get_job_details(
    project_id: str, trajectory_id: str, runner: BaseRunner = Depends(get_runner)
) -> JobDetails:
    """Get detailed information about a job.

    This endpoint returns detailed information about a job, including:
    - Basic job information (ID, status, timestamps)
    - Runner-specific details organized into sections
    - Error information if the job failed
    """
    job_details = await runner.get_job_details(project_id, trajectory_id)
    if not job_details:
        raise HTTPException(
            status_code=404,
            detail=f"Job details not found for project {project_id}, trajectory {trajectory_id}",
        )

    # Remove raw_data from the response to reduce payload size
    job_details.raw_data = None

    return job_details 