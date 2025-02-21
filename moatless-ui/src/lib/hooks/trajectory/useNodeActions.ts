import { useTrajectoryContext } from './useTrajectoryContext';
import { useRetryNode } from '../useRetryNode';
import { useNavigate } from 'react-router-dom';

export function useNodeActions(nodeId: number) {
  const { trajectory } = useTrajectoryContext();
  const retryNode = useRetryNode();
  const navigate = useNavigate();

  const handleRetry = async () => {
    await retryNode.mutateAsync({
      trajectoryId: trajectory.id,
      projectId: trajectory.project_id,
      nodeId,
    });
  };

  const handleFork = async () => {
    const response = await fetch(`/api/trajectories/${trajectory.project_id}/${trajectory.id}/fork`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ nodeId })
    });
    
    const data = await response.json();
    navigate(`/trajectory/${data.trajectoryId}`);
  };

  return {
    handleRetry,
    handleFork,
    isRetryPending: retryNode.isPending,
    canPerformActions: true
  };
} 