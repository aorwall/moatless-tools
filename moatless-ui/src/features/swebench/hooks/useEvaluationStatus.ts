import { useQuery } from '@tanstack/react-query';
import { evaluationApi } from '../api/evaluation';

export const evaluationKeys = {
  all: ['evaluation'] as const,
  details: (evaluationId: string) => [...evaluationKeys.all, evaluationId] as const,
};

export function useEvaluationStatus(evaluationId: string) {
  return useQuery({
    queryKey: evaluationKeys.details(evaluationId),
    queryFn: () => evaluationApi.getEvaluation(evaluationId),
    refetchInterval: 5000, // Poll every 5 seconds
  });
} 