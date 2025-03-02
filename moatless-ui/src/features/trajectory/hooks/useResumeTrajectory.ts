import { trajectoriesApi } from '@/lib/api/trajectories.ts';
import { ResumeTrajectoryRequest } from '@/lib/types/trajectory.ts';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';

export function useResumeTrajectory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      projectId,
      trajectoryId,
      request
    }: {
      projectId: string;
      trajectoryId: string;
      request: ResumeTrajectoryRequest
    }) => trajectoriesApi.resume(`${projectId}/${trajectoryId}`, request),

    onSuccess: (_, variables) => {
      // Invalidate the trajectory queries
      queryClient.invalidateQueries({
        queryKey: ['trajectory', variables.trajectoryId]
      });
      toast.success('Trajectory started successfully');
    },

    onError: (error) => {
      console.error('Error starting trajectory:', error);
      toast.error('Failed to start trajectory', {
        description: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    }
  });
} 