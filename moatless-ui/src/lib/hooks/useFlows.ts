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
      return await flowsApi.updateFlow(flow.id, {
        id: flow.id,
        description: flow.description,
        flow_type: flow.flow_type,
        max_expansions: flow.max_expansions,
        max_iterations: flow.max_iterations,
        max_cost: flow.max_cost,
        min_finished_nodes: flow.min_finished_nodes,
        max_finished_nodes: flow.max_finished_nodes,
        reward_threshold: flow.reward_threshold,
        max_depth: flow.max_depth,
        agent_id: flow.agent_id,
        selector: flow.selector,
        expander: flow.expander,
        value_function: flow.value_function,
        feedback_generator: flow.feedback_generator,
        discriminator: flow.discriminator,
      });
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
      return await flowsApi.createFlow(flow);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["flows"] });
    },
  });
} 