import { apiRequest } from "@/lib/api/config";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { trajectoryKeys } from "./useGetTrajectory";
import { useEffect } from "react";

// Types for structured message content
interface ToolCall {
    name: string;
    arguments: any;
}

interface CompletionInputMessage {
    role: string;
    content: string;
    tool_calls?: ToolCall[];
}

interface CompletionOutput {
    content?: string;
    tool_calls?: ToolCall[];
}

export interface CompletionResponse {
    // Parsed fields
    system_prompt?: string;
    input?: CompletionInputMessage[];
    output?: CompletionOutput;

    // Original data
    original_input?: Record<string, any>;
    original_output?: Record<string, any>;
}

// Define API functions in a centralized object
export const completionsApi = {
    getNodeCompletions: async (
        projectId: string,
        trajectoryId: string,
        nodeId: number,
    ): Promise<CompletionResponse[]> => {
        return apiRequest(
            `/trajectories/${projectId}/${trajectoryId}/completions/${nodeId}`,
            { method: "GET" }
        );
    },
    getNodeActionCompletions: async (
        projectId: string,
        trajectoryId: string,
        nodeId: number,
        actionStep: number
    ): Promise<CompletionResponse[]> => {
        return apiRequest(
            `/trajectories/${projectId}/${trajectoryId}/completions/${nodeId}/action/${actionStep}`,
            { method: "GET" }
        );
    }
};

/**
 * Hook for fetching node completions OR action completions, but not both simultaneously.
 * Prefers action completions if actionStep is provided.
 */
export function useGetNodeCompletions(
    projectId: string,
    trajectoryId: string,
    nodeId: number,
    actionStep?: number
) {
    const queryClient = useQueryClient();
    const isActionCompletion = actionStep !== undefined && actionStep !== null;

    // Define the two possible query keys
    const nodeCompletionKey = [
        ...trajectoryKeys.detail(projectId, trajectoryId),
        "completions",
        nodeId,
    ];

    const actionCompletionKey = [
        ...trajectoryKeys.detail(projectId, trajectoryId),
        "completions",
        nodeId,
        "action",
        actionStep,
    ];

    // Select the appropriate query key based on whether actionStep is provided
    const queryKey = isActionCompletion ? actionCompletionKey : nodeCompletionKey;

    // When switching between node and action completions, cancel any pending queries for the other type
    useEffect(() => {
        if (isActionCompletion) {
            queryClient.cancelQueries({ queryKey: nodeCompletionKey });
        } else {
            queryClient.cancelQueries({ queryKey: actionCompletionKey });
        }
    }, [isActionCompletion, queryClient, nodeCompletionKey, actionCompletionKey]);

    return useQuery({
        queryKey,
        queryFn: () =>
            isActionCompletion
                ? completionsApi.getNodeActionCompletions(projectId, trajectoryId, nodeId, actionStep!)
                : completionsApi.getNodeCompletions(projectId, trajectoryId, nodeId),
        retry: false,
        enabled: !!projectId && !!trajectoryId && nodeId !== undefined,
        staleTime: 30000, // Keep data fresh for 30 seconds to prevent unnecessary refetches
    });
}

/**
 * Hook specifically for fetching node completions (without action)
 */
export function useGetNodeCompletionsOnly(
    projectId: string,
    trajectoryId: string,
    nodeId: number
) {
    return useQuery({
        queryKey: [
            ...trajectoryKeys.detail(projectId, trajectoryId),
            "completions",
            nodeId,
        ],
        queryFn: () => completionsApi.getNodeCompletions(projectId, trajectoryId, nodeId),
        retry: false,
        enabled: !!projectId && !!trajectoryId && nodeId !== undefined,
        staleTime: 30000,
    });
}

/**
 * Hook specifically for fetching action completions
 */
export function useGetNodeActionCompletions(
    projectId: string,
    trajectoryId: string,
    nodeId: number,
    actionStep: number
) {
    return useQuery({
        queryKey: [
            ...trajectoryKeys.detail(projectId, trajectoryId),
            "completions",
            nodeId,
            "action",
            actionStep,
        ],
        queryFn: () => completionsApi.getNodeActionCompletions(projectId, trajectoryId, nodeId, actionStep),
        retry: false,
        enabled: !!projectId && !!trajectoryId && nodeId !== undefined && actionStep !== undefined,
        staleTime: 30000,
    });
}
