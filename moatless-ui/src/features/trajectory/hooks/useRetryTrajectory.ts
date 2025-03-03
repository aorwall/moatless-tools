import { trajectoriesApi } from "@/lib/api/trajectories.ts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { trajectoryKeys } from "./useGetTrajectory";

export function useRetryTrajectory(options?: {
  onSuccess?: (data: any, variables: { projectId: string; trajectoryId: string }) => void;
}) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      trajectoryId,
    }: {
      projectId: string;
      trajectoryId: string;
    }) => trajectoriesApi.retryTrajectory(projectId, trajectoryId),

    onSuccess: (data, variables) => {
      // Default invalidation
      queryClient.invalidateQueries({
        queryKey: trajectoryKeys.detail(
          variables.projectId,
          variables.trajectoryId,
        ),
      });

      toast.success("Trajectory retried successfully");

      // Call custom onSuccess handler if provided
      if (options?.onSuccess) {
        options.onSuccess(data, variables);
      }
    },

    onError: (error) => {
      console.error("Error retrying trajectory:", error);
      toast.error("Failed to retry trajectory", {
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
      });
    },
  });
}
