import { useNavigate } from "react-router-dom";
import { useRetryNode } from "./useRetryNode";
import { Trajectory } from "@/lib/types/trajectory";

interface UseNodeActionsProps {
  nodeId: number;
  trajectory: Trajectory;
}

export function useNodeActions({ nodeId, trajectory }: UseNodeActionsProps) {
  const retryNode = useRetryNode();
  const navigate = useNavigate();

  const handleRetry = async () => {
    if (!trajectory.id) return;

    await retryNode.mutateAsync({
      trajectoryId: trajectory.id,
      projectId: trajectory.project_id,
      nodeId,
    });
  };

  const handleFork = async () => {
    if (!trajectory.id) return;
    
    const response = await fetch(`/api/trajectories/${trajectory.id}/fork`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        nodeId
      })
    });
    
    const data = await response.json();
    navigate(`/trajectory/${data.trajectoryId}`);
  };

  return {
    handleRetry,
    handleFork,
    isRetryPending: retryNode.isPending,
    canPerformActions: !!trajectory.id
  };
} 