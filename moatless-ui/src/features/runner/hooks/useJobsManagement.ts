import { useMutation, useQueryClient } from "@tanstack/react-query";
import { runnerApi } from "../api/runnerApi";
import { runnerKeys } from "./useRunnerInfo";
import { toast } from "sonner";

/**
 * Hook for canceling a job or all jobs for a project
 */
export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      trajectoryId,
    }: {
      projectId: string;
      trajectoryId?: string;
    }) => runnerApi.cancelJob(projectId, trajectoryId),
    onSuccess: (_, variables) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: runnerKeys.info() });
      queryClient.invalidateQueries({
        queryKey: runnerKeys.jobSummary(variables.projectId),
      });

      // Show success message
      toast.success(
        variables.trajectoryId
          ? `Job ${variables.trajectoryId} canceled successfully`
          : `All jobs for project ${variables.projectId} canceled successfully`,
      );
    },
    onError: (error) => {
      toast.error(
        `Failed to cancel job: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    },
  });
}

/**
 * Hook for retrying a failed job
 */
export function useRetryJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      trajectoryId,
    }: {
      projectId: string;
      trajectoryId: string;
    }) => runnerApi.retryJob(projectId, trajectoryId),
    onSuccess: (_, variables) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: runnerKeys.info() });
      queryClient.invalidateQueries({
        queryKey: runnerKeys.jobSummary(variables.projectId),
      });

      // Show success message
      toast.success(`Job ${variables.trajectoryId} restarted successfully`);
    },
    onError: (error) => {
      toast.error(
        `Failed to retry job: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    },
  });
}
