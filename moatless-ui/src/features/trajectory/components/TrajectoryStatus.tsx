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
  RefreshCw,
  Square,
  Zap,
} from "lucide-react";
import { useEffect, useState } from "react";

interface TrajectoryStatusProps {
  trajectory: Trajectory;
  className?: string;
  handleStart?: () => Promise<void>;
  handleRetry?: () => Promise<void>;
}

export function TrajectoryStatus({
  trajectory,
  className,
  handleStart,
  handleRetry
}: TrajectoryStatusProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const queryClient = useQueryClient();
  const cancelJob = useCancelJob();

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
  const canStart =
    trajectory.status.toLowerCase() !== "running" &&
    trajectory.status.toLowerCase() !== "completed";

  // Check if the trajectory has been started or not
  const hasStarted =
    trajectory.system_status.started_at !== undefined &&
    trajectory.system_status.started_at !== null;

  // Check if the trajectory can be retried (not running)
  const canRetry = trajectory.status.toLowerCase() !== "running" && hasStarted;

  const handleStartClick = async () => {
    if (!handleStart) return;

    setIsStarting(true);
    try {
      await handleStart();
    } catch (error) {
      console.error("Error starting trajectory:", error);
    } finally {
      setIsStarting(false);
    }
  };

  const handleRetryClick = async () => {
    if (!handleRetry) return;

    setIsRetrying(true);
    try {
      await handleRetry();
    } catch (error) {
      console.error("Error retrying trajectory:", error);
    } finally {
      setIsRetrying(false);
    }
  };

  const handleCancelClick = async () => {
    cancelJob.mutate({
      projectId: trajectory.project_id,
      trajectoryId: trajectory.trajectory_id,
    });
  };

  // Periodically refetch trajectory status when running
  useEffect(() => {
    let intervalId: number | undefined;

    if (trajectory.status.toLowerCase() === "running") {
      intervalId = window.setInterval(() => {
        queryClient.invalidateQueries({
          queryKey: trajectoryKeys.detail(
            trajectory.project_id,
            trajectory.trajectory_id,
          ),
        });
      }, 5000); // Refetch every 5 seconds
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [
    trajectory.status,
    trajectory.project_id,
    trajectory.trajectory_id,
    queryClient,
  ]);

  // Determine which action buttons to show
  const getActionButtons = () => {
    if (trajectory.status.toLowerCase() === "running") {
      return (
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
      );
    }

    return (
      <div className="flex items-center gap-2 ml-auto">
        {canRetry && handleRetry && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleRetryClick}
            disabled={isRetrying}
          >
            {isRetrying ? (
              <>
                <Loader2 className="h-3 w-3 animate-spin mr-2" />
                Retrying...
              </>
            ) : (
              <>
                <RefreshCw className="h-3 w-3 mr-2" />
                Retry
              </>
            )}
          </Button>
        )}

        {canStart && handleStart && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleStartClick}
            disabled={isStarting}
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
        )}
      </div>
    );
  };

  return (
    <div
      className={cn(
        "flex items-center h-14 px-4 py-2 gap-3 border-b bg-background/50",
        className,
      )}
    >
      {/* Status Badge */}
      <Badge
        variant={trajectory?.status === "error" ? "destructive" : "default"}
        className="flex items-center gap-1 h-7 px-3"
      >
        {getStatusIcon(trajectory?.status)}
        {trajectory?.status}
      </Badge>

      {/* Main Stats Group */}
      <div className="flex items-center gap-3 text-xs">
        {/* Timing Info */}
        {hasStarted && (
          <div className="flex items-center gap-1.5 px-2 py-1 bg-muted/20 rounded-md">
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3 text-muted-foreground" />
              <span>
                Started:{" "}
                {formatDistanceToNow(
                  new Date(trajectory.system_status.started_at),
                )}{" "}
                ago
              </span>
            </div>
            {trajectory?.system_status.finished_at && (
              <>
                <Separator orientation="vertical" className="mx-1 h-3" />
                <span>
                  Finished:{" "}
                  {formatDistanceToNow(
                    new Date(trajectory?.system_status.finished_at),
                  )}{" "}
                  ago
                </span>
              </>
            )}
          </div>
        )}

        {/* Activity Info */}
        <div className="flex items-center px-2 py-1 bg-muted/20 rounded-md">
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3 text-muted-foreground" />
            <span>{trajectory.nodes?.length || 0} Actions</span>
          </div>
        </div>

        {/* Cost Info (if present) */}
        {trajectory.usage?.completion_cost && (
          <div className="flex items-center px-2 py-1 bg-muted/20 rounded-md">
            <div className="flex items-center gap-1">
              <Coins className="h-3 w-3 text-muted-foreground" />
              <span>Cost: ${trajectory.usage.completion_cost.toFixed(4)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Render action buttons on the right side */}
      {getActionButtons()}
    </div>
  );
}
