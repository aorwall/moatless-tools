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
  Bot,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Layers,
  MessageSquare,
  Package,
  Pause,
  HelpCircle,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from "@/lib/components/ui/dropdown-menu";

export type TrajectoryView = "timeline" | "chat" | "artifacts" | "timeline2" | "tree";

interface TrajectoryStatusProps {
  trajectory: Trajectory;
  className?: string;
  handleStart?: () => Promise<void>;
  handleRetry?: () => Promise<void>;
  showRightPanel?: boolean;
  onToggleRightPanel?: () => void;
  currentView?: TrajectoryView;
  onViewChange?: (view: TrajectoryView) => void;
}

export function TrajectoryStatus({
  trajectory,
  className,
  handleStart,
  handleRetry,
  showRightPanel = true,
  onToggleRightPanel,
  currentView = "timeline",
  onViewChange
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
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-success" />;
      case "running":
      case "initializing":
      case "pending":
        return <Loader2 className="h-4 w-4 animate-spin" />;
      case "canceled":
        return <Pause className="h-4 w-4 text-warning" />;
      case "failed":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case "not_found":
        return <HelpCircle className="h-4 w-4 text-muted-foreground" />;
      default:
        return null;
    }
  };

  // Get the job status or fall back to trajectory status
  const jobStatus = trajectory.job_status?.toLowerCase();

  // Check if the job is active (running, pending, or initializing)
  const isJobActive =
    jobStatus === "pending" ||
    jobStatus === "initializing" ||
    jobStatus === "running";

  // Check if the trajectory can be started
  const canStart =
    !isJobActive &&
    jobStatus !== "completed";

  // Check if the trajectory has been started or not
  const hasStarted =
    trajectory.system_status.started_at !== undefined &&
    trajectory.system_status.started_at !== null;

  // Check if the trajectory can be retried (not running)
  const canRetry = !isJobActive && hasStarted;

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

  // Periodically refetch trajectory status when job is active
  useEffect(() => {
    let intervalId: number | undefined;

    if (isJobActive) {
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
    isJobActive,
    trajectory.project_id,
    trajectory.trajectory_id,
    queryClient,
  ]);

  // Determine which action buttons to show
  const getActionButtons = () => {
    const actionButtons = [];

    // Add view selector dropdown
    if (onViewChange) {
      actionButtons.push(
        <DropdownMenu key="view-selector">
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="sm" className="mr-2">
              <Layers className="h-3 w-3 mr-2" />
              {getViewName(currentView)}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onViewChange("timeline")}>
              <Clock className="h-4 w-4 mr-2" />
              Timeline v1
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onViewChange("timeline2")}>
              <Clock className="h-4 w-4 mr-2" />
              Timeline v2
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onViewChange("chat")}>
              <MessageSquare className="h-4 w-4 mr-2" />
              Chat
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onViewChange("artifacts")}>
              <Package className="h-4 w-4 mr-2" />
              Artifacts
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onViewChange("tree")}>
              Tree
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    }

    if (onToggleRightPanel) {
      actionButtons.push(
        <Button
          key="toggle-panel"
          variant="ghost"
          size="sm"
          onClick={onToggleRightPanel}
          className="mr-2"
        >
          {showRightPanel ? (
            <>
              <ChevronRight className="h-3 w-3 mr-2" />
              Hide Details
            </>
          ) : (
            <>
              <ChevronLeft className="h-3 w-3 mr-2" />
              Show Details
            </>
          )}
        </Button>
      );
    }

    // Show cancel button if job is active
    if (isJobActive) {
      actionButtons.push(
        <Button
          key="cancel"
          variant="outline"
          size="sm"
          onClick={handleCancelClick}
          disabled={cancelJob.isPending}
          className="text-destructive hover:text-destructive"
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

      return <div className="flex items-center ml-auto">{actionButtons}</div>;
    }

    // Add retry and start buttons
    if (canRetry && handleRetry) {
      actionButtons.push(
        <Button
          key="retry"
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
      );
    }

    if (canStart && handleStart) {
      actionButtons.push(
        <Button
          key="start"
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
      );
    }

    return <div className="flex items-center gap-2 ml-auto">{actionButtons}</div>;
  };

  // Helper function to get view name for the dropdown button
  const getViewName = (view: TrajectoryView) => {
    switch (view) {
      case "timeline": return "Timeline v1";
      case "timeline2": return "Timeline v2";
      case "chat": return "Chat";
      case "artifacts": return "Artifacts";
      default: return "View";
    }
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
        variant={
          trajectory?.status === "error" || trajectory?.job_status === "failed"
            ? "destructive"
            : trajectory?.job_status === "canceled"
              ? "outline"
              : "default"
        }
        className="flex items-center gap-1 h-7 px-3"
      >
        {getStatusIcon(trajectory?.job_status || trajectory?.status)}
        {trajectory?.status}
      </Badge>

      {/* Job Status Badge (only shown when different from main status) */}
      {trajectory.job_status && trajectory.job_status.toLowerCase() !== trajectory.status.toLowerCase() && (
        <Badge
          variant={
            trajectory.job_status === "failed"
              ? "destructive"
              : trajectory.job_status === "canceled"
                ? "outline"
                : trajectory.job_status === "completed"
                  ? "default"
                  : "secondary"
          }
          className="flex items-center gap-1 h-7 px-3"
        >
          {getStatusIcon(trajectory.job_status)}
          Job: {trajectory.job_status}
        </Badge>
      )}

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

        {/* Agent and Model Info */}
        <div className="flex items-center px-2 py-1 bg-muted/20 rounded-md">
          <div className="flex items-center gap-1">
            <Bot className="h-3 w-3 text-muted-foreground" />
            <span>{trajectory.agent_id}</span>
          </div>
          <Separator orientation="vertical" className="mx-1 h-3" />
          <div className="flex items-center gap-1">
            <Cpu className="h-3 w-3 text-muted-foreground" />
            <span>{trajectory.model_id}</span>
          </div>
        </div>

        {/* Activity Info */}
        <div className="flex items-center px-2 py-1 bg-muted/20 rounded-md">
          <div className="flex items-center gap-1">
            <Zap className="h-3 w-3 text-muted-foreground" />
            <span>{trajectory.nodes?.length || 0} Actions</span>
          </div>
        </div>

        {/* Cost Info (if present) */}
        {trajectory.usage && (
          <div className="flex items-center px-2 py-1 bg-muted/20 rounded-md">
            <div className="flex items-center gap-1">
              <Coins className="h-3 w-3 text-muted-foreground" />
              {trajectory.usage.completion_cost && (
                <span>${trajectory.usage.completion_cost.toFixed(4)}</span>
              )}
            </div>

            {(trajectory.usage.prompt_tokens || trajectory.usage.cache_read_tokens) && (
              <>
                <Separator orientation="vertical" className="mx-1 h-3" />
                <div className="flex items-center gap-1">
                  <span>
                    P: {trajectory.usage.prompt_tokens || 0}
                    {trajectory.usage.cache_read_tokens ?
                      ` (${trajectory.usage.cache_read_tokens} cached)` :
                      ''}
                  </span>
                </div>
              </>
            )}

            {trajectory.usage.completion_tokens && (
              <>
                <Separator orientation="vertical" className="mx-1 h-3" />
                <div className="flex items-center gap-1">
                  <span>C: {trajectory.usage.completion_tokens}</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Render action buttons on the right side */}
      {getActionButtons()}
    </div>
  );
}
