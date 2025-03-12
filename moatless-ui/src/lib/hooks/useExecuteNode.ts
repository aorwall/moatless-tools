import { useMutation } from "@tanstack/react-query";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { toast } from "sonner";
import { Trajectory } from "@/lib/types/trajectory";

interface ExecuteNodeParams {
  nodeId: number;
  trajectory: Trajectory;
  onSuccess?: (data: any) => void;
  onError?: (error: Error | unknown) => void;
}

export const useExecuteNode = () => {
  return useMutation({
    mutationFn: ({ nodeId, trajectory }: ExecuteNodeParams) => {
      if (!nodeId) {
        throw new Error("Node ID is required to execute a node");
      }
      if (!trajectory.trajectory_id || !trajectory.project_id) {
        throw new Error(
          "Trajectory ID and Project ID must be set to execute a node",
        );
      }
      return trajectoriesApi.executeNode(
        trajectory.trajectory_id,
        trajectory.project_id,
        nodeId,
      );
    },
    onSuccess: (data, { onSuccess }) => {
      if (onSuccess) {
        onSuccess(data);
      }
    },
    onError: (error, { onError }) => {
      if (onError) {
        onError(error);
      }
      toast.error("Failed to execute node", {
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
      });
    },
  });
};
