import { useQuery } from "@tanstack/react-query";
import { runnerApi } from "../api/runnerApi";
import { JobDetails } from "../types";

// Query keys for job details
export const jobDetailsKeys = {
    all: ["jobDetails"] as const,
    detail: (projectId: string, trajectoryId: string) =>
        [...jobDetailsKeys.all, projectId, trajectoryId] as const,
};

/**
 * Hook to fetch detailed information about a job
 * @param projectId The project ID
 * @param trajectoryId The trajectory ID
 * @param enabled Whether the query should be enabled or not
 */
export function useJobDetails(
    projectId: string | undefined,
    trajectoryId: string | undefined,
    enabled = true
) {
    return useQuery<JobDetails>({
        queryKey: projectId && trajectoryId
            ? jobDetailsKeys.detail(projectId, trajectoryId)
            : jobDetailsKeys.all,
        queryFn: () => {
            if (!projectId || !trajectoryId) {
                throw new Error("Project ID and Trajectory ID are required");
            }
            return runnerApi.getJobDetails(projectId, trajectoryId);
        },
        enabled: enabled && !!projectId && !!trajectoryId, // Only fetch when both IDs are available
    });
} 