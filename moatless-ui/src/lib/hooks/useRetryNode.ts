import { useMutation, useQueryClient } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { trajectoryKeys } from "./useGetTrajectory";
import { toast } from "sonner";

interface RetryNodeParams {
  trajectoryId: string;
  nodeId: number;
  onSuccess?: () => void;
}

export const useRetryNode = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ trajectoryId, nodeId, onSuccess }: RetryNodeParams) => 
      trajectoriesApi.retryNode(trajectoryId, nodeId),
    onSuccess: (_, { trajectoryId, onSuccess }) => {
      if (onSuccess) {
        onSuccess();
      }
      queryClient.invalidateQueries({ queryKey: trajectoryKeys.detail(trajectoryId) });
    },
    onError: (error, { trajectoryId }) => {
      queryClient.invalidateQueries({ queryKey: trajectoryKeys.detail(trajectoryId) });
      toast.error("Failed to retry node", {
        description: error instanceof Error 
          ? error.message 
          : "An unexpected error occurred",
      });
    },
  });
};