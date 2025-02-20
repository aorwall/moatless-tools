import { Timeline } from "@/lib/components/trajectory/Timeline";
import { useTrajectory } from "@/lib/hooks/useTrajectory";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { useEffect } from "react";

interface TrajectoryViewerProps {
  trajectoryId: string;
}

export function TrajectoryViewer({ trajectoryId }: TrajectoryViewerProps) {
  const { data: trajectory, isLoading } = useTrajectory(trajectoryId);
  const resetInstance = useTrajectoryStore((state) => state.resetInstance);

  // Reset expansion state when trajectory changes
  useEffect(() => {
    resetInstance(trajectoryId);
  }, [trajectoryId, resetInstance]);

  if (isLoading) return <div>Loading...</div>;
  if (!trajectory) return <div>No trajectory found</div>;

  return <Timeline nodes={trajectory.nodes} instanceId={trajectoryId} isRunning={trajectory.status === "running"} />;
}
