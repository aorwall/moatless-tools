import { swebenchApi } from "@/lib/api/swebench";
import { useRealtimeQuery } from "@/lib/hooks/useRealtimeQuery";
import { Trajectory } from "@/lib/types/trajectory";
import { UseQueryOptions } from "@tanstack/react-query";
import { useState, useEffect } from "react";

export const evaluationKeys = {
    all: ["evaluation"] as const,
    lists: () => [...evaluationKeys.all, "list"] as const,
    list: (filters: Record<string, unknown>) => [...evaluationKeys.lists(), filters] as const,
    details: () => [...evaluationKeys.all, "detail"] as const,
    detail: (evaluationId: string) => [...evaluationKeys.details(), evaluationId] as const,
    instances: (evaluationId: string) => [...evaluationKeys.detail(evaluationId), "instances"] as const,
    instance: (evaluationId: string, instanceId: string) =>
        [...evaluationKeys.instances(evaluationId), instanceId] as const,
};

/**
 * Hook to fetch and subscribe to evaluation instance data
 * 
 * @param evaluationId - The evaluation ID
 * @param instanceId - The instance ID
 * @param options - Additional React Query options
 * @returns The evaluation instance data with real-time updates
 */
export function useRealtimeEvaluationInstance(
    evaluationId: string,
    instanceId: string,
    options?: Omit<UseQueryOptions<Trajectory, Error, Trajectory>, "queryKey" | "queryFn">
) {

    const result = useRealtimeQuery({
        queryKey: evaluationKeys.instance(evaluationId, instanceId),
        queryFn: () => swebenchApi.getEvaluationInstance(evaluationId, instanceId),
        subscriptionConfig: {
            projectId: evaluationId,
            trajectoryId: instanceId,
            eventFilter: {
                scopes: ["evaluation", "flow", "trajectory"],
                anyMatch: true
            }
        },
        options
    });


    return result;
} 