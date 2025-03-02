import { useQuery } from "@tanstack/react-query";
import { evaluationApi } from "../api/evaluation";

export const datasetsKeys = {
  all: ["datasets"] as const,
};

export function useDatasetsList() {
  return useQuery({
    queryKey: datasetsKeys.all,
    queryFn: () => evaluationApi.getDatasets(),
  });
}
