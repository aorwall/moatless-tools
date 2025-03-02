import { trajectoriesApi } from '@/lib/api/trajectories.ts';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { trajectoryKeys } from './useGetTrajectory';

export function useRetryTrajectory() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({
            projectId,
            trajectoryId
        }: {
            projectId: string;
            trajectoryId: string;
        }) => trajectoriesApi.retryTrajectory(projectId, trajectoryId),

        onSuccess: (_, variables) => {
            // Invalidate the trajectory queries
            queryClient.invalidateQueries({
                queryKey: trajectoryKeys.detail(variables.projectId, variables.trajectoryId)
            });
            toast.success('Trajectory retried successfully');
        },

        onError: (error) => {
            console.error('Error retrying trajectory:', error);
            toast.error('Failed to retry trajectory', {
                description: error instanceof Error ? error.message : 'Unknown error occurred'
            });
        }
    });
} 