import { trajectoryKeys } from "@/features/trajectory/hooks/useGetTrajectory";
import { trajectoriesApi } from "@/lib/api/trajectories";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

interface ResumeTrajectoryParams {
  trajectoryId: string;
  projectId: string;
  agentId: string;
  modelId: string;
  message: string;
  onSuccess?: () => void;
}

export function useResumeTrajectory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      trajectoryId,
      projectId,
      agentId,
      modelId,
      message,
      onSuccess,
    }: ResumeTrajectoryParams) =>
      trajectoriesApi.resume(trajectoryId, {
        agent_id: agentId,
        model_id: modelId,
        message: message,
      }),
    onSuccess: (_, { trajectoryId, projectId, onSuccess }) => {
      if (onSuccess) {
        onSuccess();
      }
      queryClient.invalidateQueries({
        queryKey: trajectoryKeys.detail(trajectoryId, projectId),
      });
    },
    onError: (error) => {
      toast.error("Failed to send message", {
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred",
      });
    },
  });
}
