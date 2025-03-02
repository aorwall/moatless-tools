import { trajectoriesApi } from "@/lib/api/trajectories.ts";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { trajectoryKeys } from "./useGetTrajectory";

export function useStartTrajectory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      trajectoryId,
    }: {
      projectId: string;
      trajectoryId: string;
    }) => trajectoriesApi.startTrajectory(projectId, trajectoryId),

    onSuccess: (_, variables) => {
      // Invalidate the trajectory queries
      queryClient.invalidateQueries({
        queryKey: trajectoryKeys.detail(
          variables.projectId,
          variables.trajectoryId,
        ),
      });
      toast.success("Trajectory started successfully");
    },

    onError: (error) => {
      console.error("Error starting trajectory:", error);
      toast.error("Failed to start trajectory", {
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
      });
    },
  });
}
