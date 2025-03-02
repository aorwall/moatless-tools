import { useQuery } from "@tanstack/react-query";
import { runnerApi } from "../api/runnerApi";

// Query keys for runner data
export const runnerKeys = {
  all: ["runner"] as const,
  info: () => [...runnerKeys.all, "info"] as const,
  jobs: () => [...runnerKeys.all, "jobs"] as const,
  jobSummary: (projectId: string) => [...runnerKeys.jobs(), projectId] as const,
};

/**
 * Hook to fetch runner status and active jobs
 * Refetches data every 10 seconds to keep UI updated
 */
export function useRunnerInfo() {
  return useQuery({
    queryKey: runnerKeys.info(),
    queryFn: () => runnerApi.getRunnerInfo(),
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}

/**
 * Hook to fetch job status summary for a project
 * @param projectId The project ID to fetch status for
 */
export function useJobStatusSummary(projectId: string) {
  return useQuery({
    queryKey: runnerKeys.jobSummary(projectId),
    queryFn: () => runnerApi.getJobStatusSummary(projectId),
    enabled: !!projectId,
    refetchInterval: 5000, // Refetch every 5 seconds
  });
}
