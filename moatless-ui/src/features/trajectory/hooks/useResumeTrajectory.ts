import { useMutation, useQueryClient } from '@tanstack/react-query';
import { trajectoriesApi } from '@/lib/api/trajectories.ts';
import { toast } from 'sonner';
import { ResumeTrajectoryRequest } from '@/lib/types/trajectory.ts';

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