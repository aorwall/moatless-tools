import { useQuery } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";

export const trajectoryKeys = {
  all: ["trajectory"] as const,
  detail: (trajectoryId: string) => [...trajectoryKeys.all, trajectoryId] as const,
};

export function useGetTrajectory(id: string) {
  return useQuery({
    queryKey: trajectoryKeys.detail(id),
    queryFn: () => trajectoriesApi.getTrajectory(id),
    retry: false,
  });
}
