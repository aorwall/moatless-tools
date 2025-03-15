import { apiRequest } from "@/lib/api/config";
import { useQuery } from "@tanstack/react-query";
import { trajectoryKeys } from "./useGetTrajectory";

interface MessageItem {
    role: string;
    content: string;
    [key: string]: any;
}

interface CompletionChoiceItem {
    finish_reason: string;
    index: number;
    message: {
        content: string;
        role: string;
        tool_calls: any;
        function_call: any;
        [key: string]: any;
    };
    [key: string]: any;
}

interface CompletionResponse {
    // original_input is an object with model and messages array
    original_input?: {
        model: string;
        messages: MessageItem[];
        [key: string]: any;
    };
    original_response?: {
        choices?: CompletionChoiceItem[];
        usage?: {
            prompt_tokens: number;
            completion_tokens: number;
            total_tokens: number;
            [key: string]: any;
        };
        [key: string]: any;
    };
    [key: string]: any;
}

// Define API functions in a centralized object
export const completionsApi = {
    getNodeCompletions: async (
        projectId: string,
        trajectoryId: string,
        nodeId: number
    ): Promise<CompletionResponse[]> => {
        return apiRequest(
            `/trajectories/${projectId}/${trajectoryId}/completions/${nodeId}`,
            { method: "GET" }
        );
    },
};

export function useGetNodeCompletions(
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
        // When the data is an array, return the last item as it's likely the most recent completion
        select: (data) => (Array.isArray(data) && data.length > 0 ? data[data.length - 1] : undefined),
    });
} 