import { useRealtimeQuery } from "@/lib/hooks/useRealtimeQuery";
import { useQuery } from "@tanstack/react-query";
import { evaluationApi } from "../api/evaluation";

export const evaluationKeys = {
  all: ["evaluation"] as const,
  lists: () => [...evaluationKeys.all, "list"] as const,
  list: (filters: Record<string, unknown>) => [...evaluationKeys.lists(), filters] as const,
  details: () => [...evaluationKeys.all, "detail"] as const,
  detail: (id: string) => [...evaluationKeys.details(), id] as const,
};

// Legacy hook - kept for backward compatibility
export function useEvaluation(evaluationId: string) {
  return useQuery({
    queryKey: evaluationKeys.detail(evaluationId),
    queryFn: () => evaluationApi.getEvaluation(evaluationId),
    //refetchInterval: 5000, // Poll every 5 seconds for updates
    retry: 0, // Don't retry
  });
}

// New WebSocket-enabled hook using project subscription
export function useRealtimeEvaluation(
  evaluationId: string,
  options?: any
) {
  return useRealtimeQuery({
    // Query configuration
    queryKey: evaluationKeys.detail(evaluationId),
    queryFn: () => evaluationApi.getEvaluation(evaluationId),

    // WebSocket subscription configuration
    subscriptionConfig: {
      // Use project subscription type
      projectId: evaluationId,
      subscriptionType: 'project', // Specify project subscription

      // Filter events by type and scope
      eventFilter: {
        // Filter by event scopes
        scopes: ["evaluation"],

        // Use anyMatch to match any of the scopes
        anyMatch: true
      }
    },

    // Additional React Query options
    options
  });
}
