import { useQuery } from '@tanstack/react-query';
import { instancesApi, FullSWEBenchInstance } from '../api/instances';

export function useInstance(instanceId: string | null) {
    return useQuery<FullSWEBenchInstance>({
        queryKey: ['instance', instanceId],
        queryFn: () => {
            if (!instanceId) {
                throw new Error('Instance ID is required');
            }
            return instancesApi.getInstance(instanceId);
        },
        enabled: !!instanceId,
    });
} 