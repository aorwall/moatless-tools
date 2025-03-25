import { useQuery } from "@tanstack/react-query";
import { evaluationApi } from "../api/evaluation";
import type { EvaluationListItem } from "../api/evaluation";

export const evaluationsKeys = {
  all: ["evaluations"] as const,
};

export function useEvaluationsList() {
  return useQuery<EvaluationListItem[], Error>({
    queryKey: evaluationsKeys.all,
    queryFn: () => evaluationApi.listEvaluations(),
  });
}
