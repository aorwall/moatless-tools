import { apiRequest } from "@/lib/api/config";
import { RunnerResponse, JobsStatusSummary, JobDetails } from "../types";

// API endpoints for runner management
export const runnerApi = {
  // Get runner status and all jobs
  getRunnerInfo: () => apiRequest<RunnerResponse>("/runner"),

  // Get job status summary for a project
  getJobStatusSummary: (projectId: string) =>
    apiRequest<JobsStatusSummary>(`/runner/jobs/summary/${projectId}`),

  // Cancel a job
  cancelJob: (projectId: string, trajectoryId?: string) =>
    apiRequest<void>(`/runner/jobs/${projectId}/cancel`, {
      method: "POST",
      body: JSON.stringify(trajectoryId ? { trajectory_id: trajectoryId } : {}),
    }),

  // Retry a failed job
  retryJob: (projectId: string, trajectoryId: string) =>
    apiRequest<void>(`/runner/jobs/${projectId}/${trajectoryId}/retry`, {
      method: "POST",
    }),

  // Get detailed job information
  getJobDetails: (projectId: string, trajectoryId: string) =>
    apiRequest<JobDetails>(`/runner/jobs/${projectId}/${trajectoryId}/details`),
};
