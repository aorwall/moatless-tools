import { trajectoriesApi } from "@/lib/api/trajectories";
import { useQuery } from "@tanstack/react-query";
import { trajectoryKeys } from "./useGetTrajectory";

export function useGetTrajectoryLogs(
    projectId: string,
    trajectoryId: string,
    fileName?: string
) {
    return useQuery({
        queryKey: [...trajectoryKeys.detail(projectId, trajectoryId), 'logs', fileName],
        queryFn: () => trajectoriesApi.getTrajectoryLogs(projectId, trajectoryId, fileName),
        retry: false,
    });
} 