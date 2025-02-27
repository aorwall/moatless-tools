import { useMutation, useQueryClient } from '@tanstack/react-query';
import { evaluationApi } from '../api/evaluation';
import { evaluationKeys } from './useEvaluation';

export function useProcessEvaluationResults() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (evaluationId: string) => evaluationApi.processEvaluationResults(evaluationId),
    onSuccess: (_, evaluationId) => {
      // Invalidate both the list and the specific evaluation
      queryClient.invalidateQueries({ queryKey: evaluationKeys.all });
      queryClient.invalidateQueries({ queryKey: evaluationKeys.detail(evaluationId) });
    },
  });
} 