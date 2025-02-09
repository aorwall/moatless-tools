import { useQuery } from '@tanstack/react-query';
import { evaluationApi } from '../api/evaluation';

export const evaluationKeys = {
  all: ['evaluation'] as const,
  detail: (id: string) => [...evaluationKeys.all, id] as const,
};

export function useEvaluation(evaluationId: string) {
  return useQuery({
    queryKey: evaluationKeys.detail(evaluationId),
    queryFn: () => evaluationApi.getEvaluation(evaluationId),
    //refetchInterval: 5000, // Poll every 5 seconds for updates
    retry: 0, // Don't retry
  });
} 