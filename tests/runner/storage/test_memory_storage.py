"""Tests for the in-memory job storage implementation."""

import asyncio
import unittest
from datetime import datetime
from unittest import IsolatedAsyncioTestCase

from moatless.runner.runner import JobInfo, JobStatus
from moatless.runner.storage.memory import InMemoryJobStorage


class TestInMemoryJobStorage(IsolatedAsyncioTestCase):
    """Test cases for InMemoryJobStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.storage = InMemoryJobStorage()
        # Create some test jobs
        self.test_job1 = JobInfo(
            id="project1-trajectory1",
            status=JobStatus.PENDING,
            project_id="project1",
            trajectory_id="trajectory1",
            enqueued_at=datetime.now(),
            metadata={"test": "data1"},
        )

        self.test_job2 = JobInfo(
            id="project1-trajectory2",
            status=JobStatus.RUNNING,
            project_id="project1",
            trajectory_id="trajectory2",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
            metadata={"test": "data2"},
        )

        self.test_job3 = JobInfo(
            id="project2-trajectory3",
            status=JobStatus.COMPLETED,
            project_id="project2",
            trajectory_id="trajectory3",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
            ended_at=datetime.now(),
            metadata={"test": "data3"},
        )

    async def test_add_and_get_job(self):
        """Test adding and retrieving a job."""
        # Add a job
        await self.storage.add_job(self.test_job1)

        # Retrieve the job and verify
        job = await self.storage.get_job(self.test_job1.project_id, self.test_job1.trajectory_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.id, self.test_job1.id)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.metadata, {"test": "data1"})

    async def test_get_nonexistent_job(self):
        """Test retrieving a job that doesn't exist."""
        job = await self.storage.get_job("nonexistent", "nonexistent")
        self.assertIsNone(job)

    async def test_update_job(self):
        """Test updating a job."""
        # Add a job
        await self.storage.add_job(self.test_job1)

        # Update the job
        updated_job = self.test_job1.model_copy(deep=True)
        updated_job.status = JobStatus.RUNNING
        updated_job.started_at = datetime.now()

        await self.storage.update_job(updated_job)

        # Retrieve and verify
        job = await self.storage.get_job(self.test_job1.project_id, self.test_job1.trajectory_id)
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertIsNotNone(job.started_at)

    async def test_get_jobs(self):
        """Test retrieving all jobs."""
        # Add test jobs
        await self.storage.add_job(self.test_job1)
        await self.storage.add_job(self.test_job2)
        await self.storage.add_job(self.test_job3)

        # Retrieve all jobs
        all_jobs = await self.storage.get_jobs()
        self.assertEqual(len(all_jobs), 3)

        # Retrieve jobs for a specific project
        project1_jobs = await self.storage.get_jobs("project1")
        self.assertEqual(len(project1_jobs), 2)

        project2_jobs = await self.storage.get_jobs("project2")
        self.assertEqual(len(project2_jobs), 1)
        self.assertEqual(project2_jobs[0].trajectory_id, "trajectory3")

    async def test_remove_job(self):
        """Test removing a job."""
        # Add test jobs
        await self.storage.add_job(self.test_job1)
        await self.storage.add_job(self.test_job2)

        # Remove one job
        await self.storage.remove_job(self.test_job1.project_id, self.test_job1.trajectory_id)

        # Verify it's gone
        job = await self.storage.get_job(self.test_job1.project_id, self.test_job1.trajectory_id)
        self.assertIsNone(job)

        # Verify other job still exists
        job = await self.storage.get_job(self.test_job2.project_id, self.test_job2.trajectory_id)
        self.assertIsNotNone(job)

    async def test_get_running_jobs_count(self):
        """Test counting running jobs."""
        # Add test jobs with different statuses
        await self.storage.add_job(self.test_job1)  # PENDING
        await self.storage.add_job(self.test_job2)  # RUNNING
        await self.storage.add_job(self.test_job3)  # COMPLETED

        # Test overall count
        count = await self.storage.get_running_jobs_count()
        self.assertEqual(count, 1)  # Only test_job2 is RUNNING

        # Add another running job for project1
        test_job4 = JobInfo(
            id="project1-trajectory4",
            status=JobStatus.RUNNING,
            project_id="project1",
            trajectory_id="trajectory4",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
        )
        await self.storage.add_job(test_job4)

        # Test overall count again
        count = await self.storage.get_running_jobs_count()
        self.assertEqual(count, 2)  # Now test_job2 and test_job4

        # Test count for specific project
        count = await self.storage.get_running_jobs_count("project1")
        self.assertEqual(count, 2)

        count = await self.storage.get_running_jobs_count("project2")
        self.assertEqual(count, 0)

    async def test_get_queued_jobs_count(self):
        """Test counting queued (pending) jobs."""
        # Add test jobs with different statuses
        await self.storage.add_job(self.test_job1)  # PENDING
        await self.storage.add_job(self.test_job2)  # RUNNING
        await self.storage.add_job(self.test_job3)  # COMPLETED

        # Test overall count
        count = await self.storage.get_queued_jobs_count()
        self.assertEqual(count, 1)  # Only test_job1 is PENDING

        # Add another pending job
        test_job4 = JobInfo(
            id="project2-trajectory4",
            status=JobStatus.PENDING,
            project_id="project2",
            trajectory_id="trajectory4",
            enqueued_at=datetime.now(),
        )
        await self.storage.add_job(test_job4)

        # Test overall count again
        count = await self.storage.get_queued_jobs_count()
        self.assertEqual(count, 2)

        # Test count for specific project
        count = await self.storage.get_queued_jobs_count("project1")
        self.assertEqual(count, 1)

        count = await self.storage.get_queued_jobs_count("project2")
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
