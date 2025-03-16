import { useQuery, useQueryClient } from '@tanstack/react-query';
import { trajectoriesApi } from '@/lib/api/trajectories';
import { trajectoryKeys } from './useGetTrajectory';

interface UseTreeViewProps {
    projectId: string;
    trajectoryId: string;
    enabled?: boolean;
    refetchInterval?: number; // Optional polling interval in milliseconds
}

export function useTreeView({
    projectId,
    trajectoryId,
    enabled = true,
    refetchInterval,
}: UseTreeViewProps) {
    const queryClient = useQueryClient();

    const { data, isLoading, error, refetch } = useQuery({
        queryKey: [...trajectoryKeys.detail(projectId, trajectoryId), 'tree'],
        queryFn: () => trajectoriesApi.getTreeViewData(projectId, trajectoryId),
        enabled,
        refetchInterval,
        retry: false,
    });

    const refreshTreeData = () => {
        return queryClient.invalidateQueries({
            queryKey: [...trajectoryKeys.detail(projectId, trajectoryId), 'tree']
        });
    };

    return {
        treeData: data,
        loading: isLoading,
        error: error ? error as Error : null,
        refreshTreeData,
    };
} 