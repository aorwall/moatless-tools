import { swebenchApi } from "@/lib/api/swebench";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

export function useRetryInstance(options?: {
    onSuccess?: (data: any, variables: { evaluationName: string; instanceId: string }) => void;
}) {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({
            evaluationName,
            instanceId,
        }: {
            evaluationName: string;
            instanceId: string;
        }) => swebenchApi.retryInstance(evaluationName, instanceId),
        onSuccess: (data, variables) => {
            // Default invalidation
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

            toast.success("Instance retried successfully");

            // Call custom onSuccess handler if provided
            if (options?.onSuccess) {
                options.onSuccess(data, variables);
            }
        },
        onError: (error) => {
            toast.error("Failed to retry instance", {
                description:
                    error instanceof Error ? error.message : "Unknown error occurred",
            });
        },
    });
} 