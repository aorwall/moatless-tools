import { useQuery } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { trajectoryKeys } from "./useGetTrajectory";

/**
 * Hook to fetch a single node from a trajectory
 */
export function useGetNode(
    projectId: string,
    trajectoryId: string,
    nodeId: number
) {
    return useQuery<Node>({
        queryKey: [
            ...trajectoryKeys.detail(projectId, trajectoryId),
            "node",
            nodeId,
        ],
        queryFn: () => trajectoriesApi.getNode(projectId, trajectoryId, nodeId),
        retry: false,
        enabled: Boolean(projectId) && Boolean(trajectoryId) && nodeId !== undefined,
    });
} 