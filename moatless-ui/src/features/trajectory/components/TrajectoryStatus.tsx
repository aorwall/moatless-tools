import { useCancelJob } from "@/features/runner/hooks/useJobsManagement.ts";
import { trajectoryKeys } from "@/features/trajectory/hooks/useGetTrajectory.ts";
import { Badge } from "@/lib/components/ui/badge.tsx";
import { Button } from "@/lib/components/ui/button.tsx";
import { Separator } from "@/lib/components/ui/separator.tsx";
import { Trajectory } from "@/lib/types/trajectory.ts";
import { cn } from "@/lib/utils.ts";
import { useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Coins,
  Loader2,
  Play,
  Square,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useStartTrajectory } from "../hooks/useStartTrajectory";

interface TrajectoryStatusProps {
  trajectory: Trajectory;
  className?: string;
}

export function TrajectoryStatus({ trajectory, className }: TrajectoryStatusProps) {
  const [isStarting, setIsStarting] = useState(false);
  const queryClient = useQueryClient();
  const cancelJob = useCancelJob();
  const startTrajectory = useStartTrajectory();

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case "finished":
        return <CheckCircle2 className="h-4 w-4 text-success" />;
      case "running":
        return <Loader2 className="h-4 w-4 animate-spin" />;
      default:
        return null;
    }
  };

  // Check if the trajectory can be started (not running or finished)
  const canStart = trajectory.status.toLowerCase() !== "running" && trajectory.status.toLowerCase() !== "completed";

  // Check if the trajectory has been started or not
  const hasStarted = trajectory.system_status.started_at !== undefined &&
    trajectory.system_status.started_at !== null;

  const handleStartClick = async () => {
    setIsStarting(true);
    try {
      await startTrajectory.mutateAsync({
        projectId: trajectory.project_id,
        trajectoryId: trajectory.trajectory_id
      });
    } catch (error) {
      console.error("Error starting trajectory:", error);
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancelClick = async () => {
    cancelJob.mutate({
      projectId: trajectory.project_id,
      trajectoryId: trajectory.trajectory_id
    });
  };

  // Periodically refetch trajectory status when running
  useEffect(() => {
    let intervalId: number | undefined;

    if (trajectory.status.toLowerCase() === "running") {
      intervalId = window.setInterval(() => {
        queryClient.invalidateQueries({
          queryKey: trajectoryKeys.detail(trajectory.project_id, trajectory.trajectory_id)
        });
      }, 5000); // Refetch every 5 seconds
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [trajectory.status, trajectory.project_id, trajectory.trajectory_id, queryClient]);

  const actionButton = canStart ? (
    <Button
      variant="outline"
      size="sm"
      onClick={handleStartClick}
      disabled={isStarting}
      className="ml-auto"
    >
      {isStarting ? (
        <>
          <Loader2 className="h-3 w-3 animate-spin mr-2" />
          Starting...
        </>
      ) : (
        <>
          <Play className="h-3 w-3 mr-2" />
          Start Instance
        </>
      )}
    </Button>
  ) : trajectory.status.toLowerCase() === "running" ? (
    <Button
      variant="outline"
      size="sm"
      onClick={handleCancelClick}
      disabled={cancelJob.isPending}
      className="ml-auto text-destructive hover:text-destructive"
    >
      {cancelJob.isPending ? (
        <>
          <Loader2 className="h-3 w-3 animate-spin mr-2" />
          Cancelling...
        </>
      ) : (
        <>
          <Square className="h-3 w-3 mr-2" />
          Cancel Job
        </>
      )}
    </Button>
  ) : null;

  return (
    <div className={cn("flex items-center h-14 px-4 py-2 gap-4 border-b bg-background/50", className)}>
      {/* Status Badge */}
      <Badge
        variant={trajectory?.status === "error" ? "destructive" : "default"}
        className="flex items-center gap-1 h-7 px-3"
      >
        {getStatusIcon(trajectory?.status)}
        {trajectory?.status}
      </Badge>

      {/* Timing Info */}
      {hasStarted && (
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          Started: {formatDistanceToNow(new Date(trajectory.system_status.started_at))} ago
          {trajectory?.system_status.finished_at && (
            <>
              <Separator orientation="vertical" className="mx-2 h-3" />
              Finished: {formatDistanceToNow(new Date(trajectory?.system_status.finished_at))} ago
            </>
          )}
        </div>
      )}

      {/* Stats - Token usage and activity */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <Zap className="h-3 w-3 text-muted-foreground" />
          <span>{trajectory.nodes.length} Nodes</span>
        </div>

        {trajectory.promptTokens !== undefined && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Tokens:</span>
            <span>{trajectory.promptTokens.toLocaleString()} P / {trajectory.completionTokens?.toLocaleString() || 0} C</span>
          </div>
        )}

        {trajectory.completionCost !== undefined && (
          <div className="flex items-center gap-1">
            <Coins className="h-3 w-3 text-muted-foreground" />
            <span>${trajectory.completionCost.toFixed(4)}</span>
          </div>
        )}
      </div>

      {/* Issues - conditionally show if there are any */}
      {((trajectory.failedActions !== undefined && trajectory.failedActions > 0) ||
        (trajectory.duplicatedActions !== undefined && trajectory.duplicatedActions > 0)) && (
          <div className="flex items-center gap-2 text-xs">
            {trajectory.failedActions !== undefined && trajectory.failedActions > 0 && (
              <span className="text-destructive">Failed: {trajectory.failedActions}</span>
            )}
            {trajectory.duplicatedActions !== undefined && trajectory.duplicatedActions > 0 && (
              <span className="text-warning">Duplicated: {trajectory.duplicatedActions}</span>
            )}
          </div>
        )}

      {/* Flags */}
      {trajectory.flags !== undefined && trajectory.flags.length > 0 && (
        <div className="flex items-center gap-1">
          {trajectory.flags.map((flag, index) => (
            <Badge key={index} variant="secondary" className="text-xs">
              {flag}
            </Badge>
          ))}
        </div>
      )}

      {/* Action button - pushed to the right */}
      {actionButton}
    </div>
  );
}

