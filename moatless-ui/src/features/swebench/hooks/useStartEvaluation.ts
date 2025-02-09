import { useMutation, useQueryClient } from '@tanstack/react-query';
import { evaluationApi } from '../api/evaluation';
import { evaluationKeys } from './useEvaluation';

export function useStartEvaluation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: evaluationApi.startEvaluation,
    onSuccess: (_, evaluationId) => {
      // Invalidate both the list and the specific evaluation
      queryClient.invalidateQueries({ queryKey: evaluationKeys.all });
      queryClient.invalidateQueries({ queryKey: evaluationKeys.detail(evaluationId) });
    },
  });
} 