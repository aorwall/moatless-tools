"""Tests for the Redis job storage implementation."""

import json
import unittest
from datetime import datetime
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from moatless.runner.runner import JobInfo, JobStatus
from moatless.runner.storage.redis import RedisJobStorage


class MockRedis:
    """Mock Redis implementation for testing."""
    
    def __init__(self):
        """Initialize with empty dictionaries for keys and sets."""
        self.data = {}
        self.sets = {}
        
    async def set(self, key, value):
        """Mock Redis SET command."""
        self.data[key] = value
        return True
        
    async def get(self, key):
        """Mock Redis GET command."""
        return self.data.get(key)
        
    async def delete(self, key):
        """Mock Redis DELETE command."""
        if key in self.data:
            del self.data[key]
        return True
        
    async def keys(self, pattern):
        """Mock Redis KEYS command with simple pattern matching."""
        prefix = pattern.replace("*", "")
        return [key for key in self.data.keys() if key.startswith(prefix)]
        
    async def sadd(self, key, value):
        """Mock Redis SADD command."""
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].add(value)
        return 1
        
    async def srem(self, key, value):
        """Mock Redis SREM command."""
        if key in self.sets and value in self.sets[key]:
            self.sets[key].remove(value)
            return 1
        return 0
        
    async def smembers(self, key):
        """Mock Redis SMEMBERS command."""
        return self.sets.get(key, set())
        
    async def scard(self, key):
        """Mock Redis SCARD command."""
        return len(self.sets.get(key, set()))
        
    async def sinter(self, key1, key2):
        """Mock Redis SINTER command."""
        set1 = self.sets.get(key1, set())
        set2 = self.sets.get(key2, set())
        return set1.intersection(set2)


class TestRedisJobStorage(IsolatedAsyncioTestCase):
    """Test cases for RedisJobStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Redis client
        self.mock_redis = MockRedis()
        
        # Patch the redis.from_url function to return our mock
        patcher = patch('redis.asyncio.from_url', return_value=self.mock_redis)
        self.addCleanup(patcher.stop)
        patcher.start()
        
        # Create RedisJobStorage with mocked Redis
        self.storage = RedisJobStorage(redis_url="redis://mock:6379/0")
        
        # Create some test jobs
        self.test_job1 = JobInfo(
            id="project1-trajectory1",
            status=JobStatus.PENDING,
            project_id="project1",
            trajectory_id="trajectory1",
            enqueued_at=datetime.now(),
            metadata={"test": "data1"}
        )
        
        self.test_job2 = JobInfo(
            id="project1-trajectory2",
            status=JobStatus.RUNNING,
            project_id="project1",
            trajectory_id="trajectory2",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
            metadata={"test": "data2"}
        )
        
        self.test_job3 = JobInfo(
            id="project2-trajectory3",
            status=JobStatus.COMPLETED,
            project_id="project2",
            trajectory_id="trajectory3",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
            ended_at=datetime.now(),
            metadata={"test": "data3"}
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
        
        # Check that Redis sets were updated correctly
        job_key = self.storage._get_job_key(self.test_job1.project_id, self.test_job1.trajectory_id)
        project_key = f"{self.storage._prefix}project:{self.test_job1.project_id}"
        status_key = f"{self.storage._prefix}status:{self.test_job1.status.value}"
        
        self.assertTrue(job_key in self.mock_redis.data)
        self.assertTrue(job_key in self.mock_redis.sets.get(project_key, set()))
        self.assertTrue(job_key in self.mock_redis.sets.get(status_key, set()))

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
        
        # Check that Redis sets were updated correctly
        job_key = self.storage._get_job_key(self.test_job1.project_id, self.test_job1.trajectory_id)
        old_status_key = f"{self.storage._prefix}status:{JobStatus.PENDING.value}"
        new_status_key = f"{self.storage._prefix}status:{JobStatus.RUNNING.value}"
        
        self.assertNotIn(job_key, self.mock_redis.sets.get(old_status_key, set()))
        self.assertIn(job_key, self.mock_redis.sets.get(new_status_key, set()))

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
        
        # Verify trajectories
        project1_trajectory_ids = {job.trajectory_id for job in project1_jobs}
        self.assertEqual(project1_trajectory_ids, {"trajectory1", "trajectory2"})
        
        project2_trajectory_ids = {job.trajectory_id for job in project2_jobs}
        self.assertEqual(project2_trajectory_ids, {"trajectory3"})

    async def test_remove_job(self):
        """Test removing a job."""
        # Add test jobs
        await self.storage.add_job(self.test_job1)
        await self.storage.add_job(self.test_job2)
        
        # Get keys for verification
        job_key = self.storage._get_job_key(self.test_job1.project_id, self.test_job1.trajectory_id)
        project_key = f"{self.storage._prefix}project:{self.test_job1.project_id}"
        status_key = f"{self.storage._prefix}status:{self.test_job1.status.value}"
        
        # Remove one job
        await self.storage.remove_job(self.test_job1.project_id, self.test_job1.trajectory_id)
        
        # Verify it's gone from main storage
        job = await self.storage.get_job(self.test_job1.project_id, self.test_job1.trajectory_id)
        self.assertIsNone(job)
        
        # Verify it's gone from Redis sets
        self.assertNotIn(job_key, self.mock_redis.data)
        self.assertNotIn(job_key, self.mock_redis.sets.get(project_key, set()))
        self.assertNotIn(job_key, self.mock_redis.sets.get(status_key, set()))
        
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
            started_at=datetime.now()
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
            enqueued_at=datetime.now()
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

    async def test_serialization_deserialization(self):
        """Test serialization and deserialization of job data."""
        # Create a job with all datetime fields
        job = JobInfo(
            id="test-job",
            status=JobStatus.COMPLETED,
            project_id="test",
            trajectory_id="job",
            enqueued_at=datetime.now(),
            started_at=datetime.now(),
            ended_at=datetime.now(),
            metadata={"complex": {"nested": "data"}}
        )
        
        # Serialize
        serialized = self.storage._serialize_job_info(job)
        
        # Should be valid JSON
        json_data = json.loads(serialized)
        self.assertIsInstance(json_data, dict)
        
        # Dates should be serialized as strings
        self.assertIsInstance(json_data["enqueued_at"], str)
        self.assertIsInstance(json_data["started_at"], str)
        self.assertIsInstance(json_data["ended_at"], str)
        
        # Deserialize
        deserialized = self.storage._deserialize_job_info(serialized)
        
        # Should get back a JobInfo object
        self.assertIsInstance(deserialized, JobInfo)
        
        # Dates should be datetime objects again
        self.assertIsInstance(deserialized.enqueued_at, datetime)
        self.assertIsInstance(deserialized.started_at, datetime)
        self.assertIsInstance(deserialized.ended_at, datetime)
        
        # Metadata should be preserved
        self.assertEqual(deserialized.metadata["complex"]["nested"], "data")


if __name__ == "__main__":
    unittest.main() 