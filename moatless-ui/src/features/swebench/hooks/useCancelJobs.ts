import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/api/config';
import { jobStatusKeys } from './useJobStatusSummary';
import { evaluationKeys } from './useEvaluation';

export interface CancelJobsResponse {
  project_id: string;
  canceled_queued_jobs: number;
  canceled_running_jobs: number;
  message: string;
}

const cancelJobs = async (evaluationId: string): Promise<CancelJobsResponse> => {
  return apiRequest(`/swebench/evaluations/${evaluationId}/jobs/cancel`, {
    method: 'POST',
  });
};

export function useCancelJobs() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: cancelJobs,
    onSuccess: (_, evaluationId) => {
      // Invalidate job status queries
      queryClient.invalidateQueries({ queryKey: jobStatusKeys.detail(evaluationId) });
      
      // Invalidate evaluation queries to reflect updated status
      queryClient.invalidateQueries({ queryKey: evaluationKeys.detail(evaluationId) });
    },
  });
} 