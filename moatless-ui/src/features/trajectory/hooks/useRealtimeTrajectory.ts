import { trajectoriesApi } from "@/lib/api/trajectories";
import { useRealtimeQuery } from "@/lib/hooks/useRealtimeQuery";
import { Trajectory } from "@/lib/types/trajectory";
import { UseQueryOptions } from "@tanstack/react-query";

/**
 * Query keys for trajectory data
 */
export const trajectoryKeys = {
    all: ["trajectory"] as const,
    lists: () => [...trajectoryKeys.all, "list"] as const,
    list: (filters: Record<string, unknown>) => [...trajectoryKeys.lists(), filters] as const,
    details: () => [...trajectoryKeys.all, "detail"] as const,
    detail: (projectId: string, trajectoryId: string) =>
        [...trajectoryKeys.details(), projectId, trajectoryId] as const,
};

/**
 * Hook to fetch and subscribe to trajectory data
 * 
 * @param projectId - The project ID
 * @param trajectoryId - The trajectory ID
 * @param options - Additional React Query options
 * @returns The trajectory data with real-time updates
 */
export function useRealtimeTrajectory(
    projectId: string,
    trajectoryId: string,
    options?: Omit<UseQueryOptions<Trajectory, Error, Trajectory>, "queryKey" | "queryFn">
) {
    return useRealtimeQuery({
        queryKey: trajectoryKeys.detail(projectId, trajectoryId),
        queryFn: () => trajectoriesApi.getTrajectory(projectId, trajectoryId),
        subscriptionConfig: {
            projectId,
            trajectoryId,
            eventFilter: {
                scopes: ["flow", "trajectory"],
                anyMatch: true
            }
        },
        options
    });
} 