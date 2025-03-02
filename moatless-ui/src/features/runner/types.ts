import { z } from "zod";

// Enum definitions matching backend
export enum JobStatus {
  PENDING = "pending",
  QUEUED = "queued",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELED = "canceled",
}

export enum RunnerStatus {
  RUNNING = "running",
  STOPPED = "stopped",
  ERROR = "error",
}

// Schema definitions
export const JobInfoSchema = z.object({
  id: z.string(),
  status: z.nativeEnum(JobStatus),
  enqueued_at: z.string().datetime().nullable(),
  started_at: z.string().datetime().nullable(),
  ended_at: z.string().datetime().nullable(),
  exc_info: z.string().nullable(),
});

export const RunnerInfoSchema = z.object({
  runner_type: z.string(),
  status: z.nativeEnum(RunnerStatus),
  data: z.record(z.any()),
});

export const JobsStatusSummarySchema = z.object({
  project_id: z.string(),
  total_jobs: z.number(),
  queued_jobs: z.number(),
  running_jobs: z.number(),
  completed_jobs: z.number(),
  failed_jobs: z.number(),
  canceled_jobs: z.number(),
  pending_jobs: z.number(),
  job_ids: z.record(z.array(z.string())).optional(),
});

export const RunnerResponseSchema = z.object({
  info: RunnerInfoSchema,
  jobs: z.array(JobInfoSchema),
});

// Type definitions
export type JobInfo = z.infer<typeof JobInfoSchema>;
export type RunnerInfo = z.infer<typeof RunnerInfoSchema>;
export type JobsStatusSummary = z.infer<typeof JobsStatusSummarySchema>;
export type RunnerResponse = z.infer<typeof RunnerResponseSchema>;
