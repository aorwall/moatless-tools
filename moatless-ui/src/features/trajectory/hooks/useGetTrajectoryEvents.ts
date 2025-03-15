import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api/config";
import { TrajectoryEvent } from "@/lib/types/trajectory";

// Define query keys for trajectory events
export const trajectoryEventsKeys = {
    all: ["trajectory-events"] as const,
    lists: () => [...trajectoryEventsKeys.all, "list"] as const,
    list: (projectId: string, trajectoryId: string) =>
        [...trajectoryEventsKeys.lists(), projectId, trajectoryId] as const,
};

// API function to get trajectory events
export const getTrajectoryEvents = async (
    projectId: string,
    trajectoryId: string
): Promise<TrajectoryEvent[]> => {
    return apiRequest(`/trajectories/${projectId}/${trajectoryId}/events`);
};

// Hook to fetch trajectory events
export function useGetTrajectoryEvents(
    projectId: string,
    trajectoryId: string,
    status?: string
) {
    const isRunning = status === "running";

    return useQuery({
        queryKey: trajectoryEventsKeys.list(projectId, trajectoryId),
        queryFn: () => getTrajectoryEvents(projectId, trajectoryId),
        refetchInterval: isRunning ? 5000 : false, // Only poll when trajectory is running
        enabled: !!projectId && !!trajectoryId,
    });
} 