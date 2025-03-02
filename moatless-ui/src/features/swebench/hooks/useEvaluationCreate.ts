import { useMutation } from "@tanstack/react-query";
import { evaluationApi, EvaluationRequest } from "../api/evaluation";

export function useEvaluationCreate() {
  return useMutation({
    mutationFn: (data: EvaluationRequest) =>
      evaluationApi.createEvaluation(data),
  });
}
