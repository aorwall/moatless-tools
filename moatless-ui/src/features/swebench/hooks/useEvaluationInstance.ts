import { useQuery } from "@tanstack/react-query";
import { swebenchApi } from "@/lib/api/swebench";
import { Trajectory } from "@/lib/types/trajectory";

export function useEvaluationInstance(
  evaluationId: string,
  instanceId: string,
) {
  return useQuery<Trajectory>({
    queryKey: ["evaluation", evaluationId, "instance", instanceId],
    queryFn: () => swebenchApi.getEvaluationInstance(evaluationId, instanceId),
    enabled: !!evaluationId && !!instanceId,
  });
}
