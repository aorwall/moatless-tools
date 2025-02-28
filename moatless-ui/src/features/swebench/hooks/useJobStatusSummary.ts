import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '@/lib/api/config';

export interface JobStatusSummary {
  project_id: string;
  total_jobs: number;
  queued_jobs: number;
  running_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  canceled_jobs: number;
  pending_jobs: number;
}

const fetchJobStatusSummary = async (evaluationId: string): Promise<JobStatusSummary> => {
  return apiRequest(`/swebench/evaluations/${evaluationId}/jobs/status`, {
    method: 'GET',
  });
};

export const jobStatusKeys = {
  all: ['job-status'] as const,
  detail: (evaluationId: string) => [...jobStatusKeys.all, evaluationId] as const,
};

export function useJobStatusSummary(evaluationId: string) {
  return useQuery({
    queryKey: jobStatusKeys.detail(evaluationId),
    queryFn: () => fetchJobStatusSummary(evaluationId),
    refetchInterval: 5000, // Refetch every 5 seconds
  });
} 