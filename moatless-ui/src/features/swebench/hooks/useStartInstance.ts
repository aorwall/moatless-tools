import { useMutation, useQueryClient } from "@tanstack/react-query";
import { swebenchApi } from "@/lib/api/swebench";
import { toast } from "sonner";

export function useStartInstance() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      evaluationName,
      instanceId,
    }: {
      evaluationName: string;
      instanceId: string;
    }) => swebenchApi.startInstance(evaluationName, instanceId),
    onSuccess: (_, variables) => {
      // Invalidate the specific evaluation and instance
      queryClient.invalidateQueries({
        queryKey: ["evaluation", variables.evaluationName],
      });
      queryClient.invalidateQueries({
        queryKey: [
          "evaluation",
          variables.evaluationName,
          "instance",
          variables.instanceId,
        ],
      });
      toast.success("Instance started successfully");
    },
    onError: (error) => {
      toast.error("Failed to start instance", {
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
      });
    },
  });
}
