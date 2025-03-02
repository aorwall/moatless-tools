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
import { useRetryTrajectory } from "../hooks/useRetryTrajectory";
import { useStartTrajectory } from "../hooks/useStartTrajectory";

interface TrajectoryStatusProps {
  trajectory: Trajectory;
  className?: string;
}

export function TrajectoryStatus({
  trajectory,
  className,
}: TrajectoryStatusProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const queryClient = useQueryClient();
  const cancelJob = useCancelJob();
  const startTrajectory = useStartTrajectory();
  const retryTrajectory = useRetryTrajectory();

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
    setIsStarting(true);
    try {
      await startTrajectory.mutateAsync({
        projectId: trajectory.project_id,
        trajectoryId: trajectory.trajectory_id,
      });
    } catch (error) {
      console.error("Error starting trajectory:", error);
    } finally {
      setIsStarting(false);
    }
  };

  const handleRetryClick = async () => {
    setIsRetrying(true);
    try {
      await retryTrajectory.mutateAsync({
        projectId: trajectory.project_id,
        trajectoryId: trajectory.trajectory_id,
      });
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
        {canRetry && (
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

        {canStart && (
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
            <span>{trajectory.nodes.length} Nodes</span>
          </div>
        </div>

        {/* Token Usage Group */}
        {trajectory.usage && (
          <div className="flex items-center gap-2 px-2 py-1 bg-muted/20 rounded-md">
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">Prompt:</span>
              <span className="font-medium">
                {trajectory.usage.prompt_tokens?.toLocaleString() || 0}
              </span>
            </div>

            <Separator orientation="vertical" className="h-3" />

            <div className="flex items-center gap-1">
              <span className="text-muted-foreground hidden sm:inline">
                Completion:
              </span>
              <span className="font-medium">
                {trajectory.usage.completion_tokens?.toLocaleString() || 0}
              </span>
            </div>

            {trajectory.usage.cache_read_tokens &&
              trajectory.usage.cache_read_tokens > 0 && (
                <>
                  <Separator orientation="vertical" className="h-3" />
                  <div className="flex items-center gap-1">
                    <span className="text-muted-foreground hidden md:inline">
                      Cached:
                    </span>
                    <span className="font-medium">
                      {trajectory.usage.cache_read_tokens.toLocaleString()}
                    </span>
                  </div>
                </>
              )}

            {trajectory.usage.cache_write_tokens &&
              trajectory.usage.cache_write_tokens > 0 && (
                <>
                  <Separator orientation="vertical" className="h-3" />
                  <div className="flex items-center gap-1">
                    <span className="text-muted-foreground hidden md:inline">
                      Cache Write:
                    </span>
                    <span className="font-medium">
                      {trajectory.usage.cache_write_tokens.toLocaleString()}
                    </span>
                  </div>
                </>
              )}

            <Separator orientation="vertical" className="h-3" />

            <div className="flex items-center gap-1">
              <Coins className="h-3 w-3 text-muted-foreground" />
              <span className="font-medium">
                ${trajectory.usage.completion_cost?.toFixed(4) || 0}
              </span>
            </div>
          </div>
        )}

        {/* Flags */}
        {trajectory.flags !== undefined && trajectory.flags.length > 0 && (
          <div className="flex items-center gap-1 px-2 py-1 bg-muted/20 rounded-md">
            {trajectory.flags.map((flag, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {flag}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="ml-auto">{getActionButtons()}</div>
    </div>
  );
}
