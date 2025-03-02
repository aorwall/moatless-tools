import { useMutation, useQueryClient } from '@tanstack/react-query';
import { trajectoriesApi } from '@/lib/api/trajectories.ts';
import { toast } from 'sonner';

export function useStartTrajectory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ 
      projectId, 
      trajectoryId 
    }: { 
      projectId: string; 
      trajectoryId: string;
    }) => trajectoriesApi.start(projectId, trajectoryId),
    
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