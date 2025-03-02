import { Button } from "@/lib/components/ui/button";
import { Play, Copy, RefreshCw, AlertCircle, StopCircle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";
import { Evaluation } from "../api/evaluation";
import { useRunnerStatus } from "@/lib/hooks/useRunnerStatus";
import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";
import { JobStatusBar } from "./JobStatusBar";
import { useJobStatusSummary } from "../hooks/useJobStatusSummary";
import { useCancelJobs } from "../hooks/useCancelJobs";
import { toast } from "sonner";

interface EvaluationToolbarProps {
  evaluation: Evaluation;
  onClone: () => void;
  onStart: () => void;
  onProcess: () => void;
  isStartPending: boolean;
  isClonePending: boolean;
  isProcessPending: boolean;
  canStart: boolean;
}

export function EvaluationToolbar({
  evaluation,
  onClone,
  onStart,
  onProcess,
  isStartPending,
  isClonePending,
  isProcessPending,
  canStart,
}: EvaluationToolbarProps) {
  const { data: runnerStatus, isLoading: isRunnerStatusLoading } =
    useRunnerStatus();
  const { data: jobStatus, isLoading: isJobStatusLoading } =
    useJobStatusSummary(evaluation.evaluation_name);
  const { mutate: cancelJobs, isPending: isCancelPending } = useCancelJobs();

  const activeWorkers = runnerStatus?.info?.data?.active_workers ?? 0;
  const totalWorkers = runnerStatus?.info?.data?.total_workers ?? 0;
  const runnerStatusText = runnerStatus?.info?.status ?? "unknown";
  const hasActiveWorkers = activeWorkers > 0;
  const workersRunning = !isRunnerStatusLoading && hasActiveWorkers;

  // Check if there are any active jobs
  const activeJobs =
    (jobStatus?.queued_jobs || 0) + (jobStatus?.running_jobs || 0);
  const hasActiveJobs = activeJobs > 0;

  // Show job status if starting or if there are active jobs
  const showJobStatus =
    isStartPending || hasActiveJobs || evaluation.status === "running";

  // Disable start button if there are active jobs or other conditions
  const startDisabled =
    isStartPending || !canStart || !workersRunning || hasActiveJobs;

  // Check if evaluation is completed
  const isCompleted = evaluation.status === "completed";

  const handleCancelJobs = () => {
    cancelJobs(evaluation.evaluation_name, {
      onSuccess: (data) => {
        toast.success(
          `Canceled ${data.canceled_queued_jobs + data.canceled_running_jobs} jobs`,
        );
      },
      onError: (error) => {
        toast.error(`Failed to cancel jobs: ${error.message}`);
      },
    });
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                onClick={onClone}
                disabled={isClonePending}
              >
                <Copy className="h-4 w-4 mr-1" />
                Clone
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Clone this evaluation</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>

        {/* Only show Sync Results button if evaluation is not completed */}
        {!isCompleted && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onProcess}
                  disabled={isProcessPending}
                >
                  <RefreshCw className="h-4 w-4 mr-1" />
                  Sync Results
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Synchronize evaluation results</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {hasActiveJobs ? (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={handleCancelJobs}
                  disabled={isCancelPending}
                >
                  <StopCircle className="h-4 w-4 mr-1" />
                  Cancel Jobs
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Cancel all running and queued jobs</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ) : (
          canStart && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button size="sm" onClick={onStart} disabled={startDisabled}>
                    {!workersRunning && (
                      <AlertCircle className="h-4 w-4 mr-1" />
                    )}
                    {workersRunning && <Play className="h-4 w-4 mr-1" />}
                    Start
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  {!workersRunning ? (
                    <p>
                      Workers not running. Start workers to enable evaluation.
                    </p>
                  ) : hasActiveJobs ? (
                    <p>
                      Cannot start while jobs are running. Cancel jobs first.
                    </p>
                  ) : (
                    <p>Start or restart evaluation</p>
                  )}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )
        )}
      </div>

      {/* Status information */}
      <div className="flex items-center gap-2">
        {!workersRunning && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div
                  className={cn(
                    "text-xs font-medium px-2 py-1 rounded flex items-center gap-1",
                    "text-amber-700 bg-amber-50 border border-amber-200",
                  )}
                >
                  <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                  Runner service is not active. Start workers to enable
                  evaluations.
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Status: {runnerStatusText}</p>
                <p>
                  No active workers available. Start the runner service to
                  enable evaluations.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}

        {/* Show job status if starting, running, or there are active jobs */}
        {!isJobStatusLoading && (showJobStatus || isStartPending) && (
          <div>
            {isStartPending && !jobStatus ? (
              <div
                className={cn(
                  "text-xs font-medium px-2 py-1 rounded flex items-center gap-1",
                  "text-blue-700 bg-blue-50 border border-blue-200",
                )}
              >
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                Starting jobs...
              </div>
            ) : (
              <JobStatusBar evaluationId={evaluation.evaluation_name} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
