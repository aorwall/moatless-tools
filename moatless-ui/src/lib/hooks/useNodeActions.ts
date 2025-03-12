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
    if (!trajectory.trajectory_id) return;

    await retryNode.mutateAsync({
      trajectoryId: trajectory.trajectory_id,
      projectId: trajectory.project_id,
      nodeId,
    });
  };

  const handleFork = async () => {
    if (!trajectory.trajectory_id) return;

    const response = await fetch(`/api/trajectories/${trajectory.trajectory_id}/fork`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        nodeId,
      }),
    });

    const data = await response.json();
    navigate(`/trajectory/${data.trajectoryId}`);
  };

  return {
    handleRetry,
    handleFork,
    isRetryPending: retryNode.isPending,
    canPerformActions: !!trajectory.trajectory_id,
  };
}
