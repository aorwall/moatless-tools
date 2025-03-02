import { trajectoryKeys } from "@/features/trajectory/hooks/useGetTrajectory";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

interface RetryNodeParams {
  trajectoryId: string;
  projectId: string;
  nodeId: number;
  onSuccess?: () => void;
}

export const useRetryNode = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ trajectoryId, projectId, nodeId, onSuccess }: RetryNodeParams) =>
      trajectoriesApi.retryNode(trajectoryId, projectId, nodeId),
    onSuccess: (_, { trajectoryId, onSuccess }) => {
      if (onSuccess) {
        onSuccess();
      }
      queryClient.invalidateQueries({ queryKey: trajectoryKeys.detail(trajectoryId, projectId) });
    },
    onError: (error, { trajectoryId, projectId }) => {
      queryClient.invalidateQueries({ queryKey: trajectoryKeys.detail(trajectoryId, projectId) });
      toast.error("Failed to retry node", {
        description: error instanceof Error
          ? error.message
          : "An unexpected error occurred",
      });
    },
  });
};