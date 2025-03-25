import { Button } from "@/lib/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";
import { useRunnerStatus } from "@/lib/hooks/useRunnerStatus";
import { AlertCircle, Copy, Play, RefreshCw, StopCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Evaluation } from "../api/evaluation";
import { useCancelJobs } from "../hooks/useCancelJobs";
import { useCloneEvaluation } from "../hooks/useCloneEvaluation";
import { useProcessEvaluationResults } from "../hooks/useProcessEvaluationResults";
import { useStartEvaluation } from "../hooks/useStartEvaluation";

interface EvaluationToolbarProps {
  evaluation: Evaluation;
  canStart: boolean;
}

export function EvaluationToolbar({
  evaluation,
  canStart,
}: EvaluationToolbarProps) {
  const navigate = useNavigate();
  const { data: runnerStatus, isLoading: isRunnerStatusLoading } = useRunnerStatus();
  const { mutate: cancelJobs, isPending: isCancelPending } = useCancelJobs();
  const { mutate: startEvaluation, isPending: isStartPending } = useStartEvaluation();
  const { mutate: cloneEvaluation, isPending: isClonePending } = useCloneEvaluation();
  const { mutate: processResults, isPending: isProcessPending } = useProcessEvaluationResults();

  const activeWorkers = runnerStatus?.info?.data?.active_workers ?? 0;
  const hasActiveWorkers = activeWorkers > 0;
  const workersRunning = !isRunnerStatusLoading && hasActiveWorkers;

  // Get active jobs directly from evaluation status
  const hasActiveJobs = evaluation.status === "running";

  // Disable start button if there are active jobs or other conditions
  const startDisabled = isStartPending || !canStart || !workersRunning || hasActiveJobs;

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

  const handleStart = () => {
    startEvaluation(
      {
        evaluationId: evaluation.evaluation_name,
        numConcurrentInstances: 1, // Default to 1 concurrent instance
      },
      {
        onSuccess: () => {
          toast.success("Evaluation started successfully");
        },
        onError: (error) => {
          toast.error(`Failed to start evaluation: ${error.message}`);
        },
      },
    );
  };

  const handleClone = () => {
    cloneEvaluation(evaluation.evaluation_name, {
      onSuccess: (data) => {
        toast.success("Evaluation cloned successfully");
        navigate(`/swebench/evaluation/${data.evaluation_name}`);
      },
      onError: (error) => {
        toast.error(`Failed to clone evaluation: ${error.message}`);
      },
    });
  };

  const handleProcessResults = () => {
    processResults(evaluation.evaluation_name, {
      onSuccess: () => {
        toast.success("Evaluation results synchronized successfully");
      },
      onError: (error) => {
        toast.error(
          `Failed to synchronize evaluation results: ${error.message}`,
        );
      },
    });
  };

  return (
    <div className="flex items-center gap-2">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="sm"
              variant="outline"
              onClick={handleClone}
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
      {true && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                onClick={handleProcessResults}
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
                <Button size="sm" onClick={handleStart}>
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
  );
}
