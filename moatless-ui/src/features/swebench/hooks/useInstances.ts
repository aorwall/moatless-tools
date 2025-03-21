import { useQuery } from "@tanstack/react-query";
import { instancesApi, InstancesQueryParams, InstancesResponse } from "../api/instances";

export const instanceKeys = {
    all: ['instances'] as const,
    list: (params: InstancesQueryParams) => [...instanceKeys.all, 'list', JSON.stringify(params)] as const,
};

export function useInstances(params: InstancesQueryParams = {}) {
    return useQuery<InstancesResponse>({
        queryKey: instanceKeys.list(params),
        queryFn: () => instancesApi.getInstances(params),
        refetchOnWindowFocus: false,
        refetchOnMount: true,
        staleTime: 0, // Always consider data stale to force refetch
    });
} 