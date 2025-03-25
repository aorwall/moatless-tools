import { z } from "zod";

// Enum definitions matching backend
export enum JobStatus {
  PENDING = "pending",
  INITIALIZING = "initializing",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELED = "canceled",
  NOT_FOUND = "not_found"
}

export enum RunnerStatus {
  RUNNING = "running",
  STOPPED = "stopped",
  ERROR = "error"
}

// Schema definitions
export const JobInfoSchema = z.object({
  id: z.string(),
  status: z.nativeEnum(JobStatus),
  project_id: z.string().nullable(),
  trajectory_id: z.string().nullable(),
  enqueued_at: z.string().datetime().nullable(),
  started_at: z.string().datetime().nullable(),
  ended_at: z.string().datetime().nullable(),
  metadata: z.record(z.any()).nullable(),
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
export type RunnerResponse = z.infer<typeof RunnerResponseSchema>;

export type RunnerStats = {
  runner_type: string;
  status: string;
  active_workers: number;
  total_workers: number;
  pending_jobs: number;
  initializing_jobs: number;
  running_jobs: number;
  total_jobs: number;
};

export type JobsStatusSummary = {
  project_id: string;
  total_jobs: number;
  pending_jobs: number;
  initializing_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  canceled_jobs: number;
  job_ids: {
    pending: string[];
    initializing: string[];
    running: string[];
    completed: string[];
    failed: string[];
    canceled: string[];
  };
};

export interface JobDetailSection {
  name: string;
  display_name: string;
  data?: Record<string, any>;
  items?: Record<string, any>[];
}

export interface JobDetails {
  id: string;
  status: JobStatus;
  project_id?: string;
  trajectory_id?: string;
  enqueued_at?: string;
  started_at?: string;
  ended_at?: string;
  sections: JobDetailSection[];
  error?: string;
}
