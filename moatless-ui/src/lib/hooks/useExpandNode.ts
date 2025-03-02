import { trajectoryKeys } from "@/features/trajectory/hooks/useGetTrajectory";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

interface ExpandNodeParams {
  trajectoryId: string;
  nodeId: number;
  agent_id: string;
  model_id: string;
  message: string;
  attachments?: { name: string; data: string }[];
  onSuccess?: () => void;
}

export const useExpandNode = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      trajectoryId,
      nodeId,
      onSuccess,
      ...params
    }: ExpandNodeParams) =>
      trajectoriesApi.expandNode(trajectoryId, nodeId, params),
    onSuccess: (_, { trajectoryId, onSuccess }) => {
      if (onSuccess) {
        onSuccess();
      }
      queryClient.invalidateQueries({
        queryKey: trajectoryKeys.detail(trajectoryId),
      });
    },
    onError: (error, { trajectoryId }) => {
      queryClient.invalidateQueries({
        queryKey: trajectoryKeys.detail(trajectoryId),
      });
      toast.error("Failed to expand node", {
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
      });
    },
  });
};
