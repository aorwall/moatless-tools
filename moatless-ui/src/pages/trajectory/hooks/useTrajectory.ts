import { useQuery } from "@tanstack/react-query";
import { trajectoryApi } from "@/lib/api/trajectory";

export const trajectoryKeys = {
  all: ["trajectories"] as const,
  single: (id: string) => [...trajectoryKeys.all, id] as const,
};

export function useTrajectory(id: string) {
  return useQuery({
    queryKey: trajectoryKeys.single(id),
    queryFn: () => trajectoryApi.getTrajectory(id),
  });
}
