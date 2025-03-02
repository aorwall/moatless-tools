import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { flowsApi } from "@/lib/api/flows";
import type { FlowConfig } from "@/lib/types/flow";

export function useFlows() {
  return useQuery({
    queryKey: ["flows"],
    queryFn: async () => {
      return await flowsApi.getFlows();
    },
  });
}

export function useFlow(id: string) {
  return useQuery({
    queryKey: ["flows", id],
    queryFn: async () => {
      return await flowsApi.getFlow(id);
    },
    enabled: !!id,
  });
}

export function useUpdateFlow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (flow: FlowConfig) => {
      // Make a copy of the flow to avoid modifying the original
      const flowToSend = { ...flow };

      // Ensure artifact_handlers is properly formatted
      if (
        flowToSend.artifact_handlers &&
        Array.isArray(flowToSend.artifact_handlers)
      ) {
        // Format each artifact handler to include its class name and properties
        flowToSend.artifact_handlers = flowToSend.artifact_handlers.map(
          (handler) => {
            if (typeof handler === "object" && handler.artifact_handler_class) {
              return {
                ...handler,
              };
            }
            return handler;
          },
        );
      }

      return await flowsApi.updateFlow(flowToSend.id, flowToSend);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["flows"] });
    },
  });
}

export function useCreateFlow() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (flow: Omit<FlowConfig, "id"> & { id: string }) => {
      // Make a copy of the flow to avoid modifying the original
      const flowToSend = { ...flow };

      // Ensure artifact_handlers is properly formatted
      if (
        flowToSend.artifact_handlers &&
        Array.isArray(flowToSend.artifact_handlers)
      ) {
        // Format each artifact handler to include its class name and properties
        flowToSend.artifact_handlers = flowToSend.artifact_handlers.map(
          (handler) => {
            if (typeof handler === "object" && handler.artifact_handler_class) {
              return {
                ...handler,
              };
            }
            return handler;
          },
        );
      }

      return await flowsApi.createFlow(flowToSend);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["flows"] });
    },
  });
}
