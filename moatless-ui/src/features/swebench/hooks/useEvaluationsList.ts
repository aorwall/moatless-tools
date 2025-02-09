import { useQuery } from '@tanstack/react-query';
import { evaluationApi } from '../api/evaluation';

export const evaluationsKeys = {
  all: ['evaluations'] as const,
};

export function useEvaluationsList() {
  return useQuery({
    queryKey: evaluationsKeys.all,
    queryFn: () => evaluationApi.listEvaluations(),
    refetchInterval: 5000, // Poll every 5 seconds
  });
} 