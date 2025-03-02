import { useQuery } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";

export const trajectoryKeys = {
  all: ["trajectory"] as const,
  detail: (projectId: string, trajectoryId: string) =>
    [...trajectoryKeys.all, projectId, trajectoryId] as const,
};

export function useGetTrajectory(projectId: string, trajectoryId: string) {
  return useQuery({
    queryKey: trajectoryKeys.detail(projectId, trajectoryId),
    queryFn: () => trajectoriesApi.getTrajectory(projectId, trajectoryId),
    retry: false,
  });
}
