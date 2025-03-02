import { useMutation, useQueryClient } from "@tanstack/react-query";
import { evaluationApi } from "../api/evaluation";
import { evaluationKeys } from "./useEvaluation";

export function useCloneEvaluation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: evaluationApi.cloneEvaluation,
    onSuccess: (data) => {
      // Invalidate both the list and the specific evaluation
      queryClient.invalidateQueries({ queryKey: evaluationKeys.all });
      return data;
    },
  });
}
