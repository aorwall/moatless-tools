import { useRunnerStats } from "../hooks/useRunnerStats";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Loader2, AlertCircle, Server } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";
import { JobStatus, RunnerStatus } from "../types";

export function RunnerStatusBar() {
  const { data, isLoading, error } = useRunnerStats();

  if (isLoading) {
    return (
      <div className="flex items-center text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
        Loading runner status...
      </div>
    );
  }

  if (error || !data) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              to="/runner"
              className="flex items-center text-xs text-red-500"
            >
              <AlertCircle className="h-3 w-3 mr-1" />
              Runner error
            </Link>
          </TooltipTrigger>
          <TooltipContent>
            <p>Error loading runner status. Click to view details.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  const runnerIsUp = data.status === RunnerStatus.RUNNING;
  const activeWorkers = data.active_workers || 0;

  // Sum job counts
  const pendingJobs = data.pending_jobs;
  const initializingJobs = data.initializing_jobs;
  const runningJobs = data.running_jobs;
  const totalJobs = data.total_jobs;

  // Calculate different job group totals
  const inProgressJobs = initializingJobs + runningJobs;
  const activeJobs = pendingJobs + initializingJobs + runningJobs;
  const hasOtherJobs = totalJobs > 0 && activeJobs === 0;

  if (!runnerIsUp && totalJobs === 0) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              to="/runner"
              className="flex items-center text-xs text-orange-500"
            >
              <Server className="h-3 w-3 mr-1" />
              Runner stopped
            </Link>
          </TooltipTrigger>
          <TooltipContent>
            <p>The runner is currently stopped. No jobs can be processed.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Link
            to="/runner"
            className={cn(
              "text-xs font-medium px-2 py-1 rounded flex items-center gap-1",
              runnerIsUp
                ? totalJobs > 0
                  ? "text-blue-700 bg-blue-50 border border-blue-200"
                  : "text-green-700 bg-green-50 border border-green-200"
                : "text-orange-700 bg-orange-50 border border-orange-200",
            )}
          >
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                runnerIsUp
                  ? totalJobs > 0
                    ? "bg-blue-500 animate-pulse"
                    : "bg-green-500"
                  : "bg-orange-500",
              )}
            ></span>
            {runnerIsUp ? (
              <>
                {activeJobs > 0 ? (
                  <>
                    {pendingJobs > 0 && (
                      <span className="font-medium">{pendingJobs} pending</span>
                    )}
                    {pendingJobs > 0 && inProgressJobs > 0 && (
                      <span className="mx-0.5">·</span>
                    )}
                    {inProgressJobs > 0 && (
                      <span className="font-medium">{inProgressJobs} running</span>
                    )}
                  </>
                ) : hasOtherJobs ? (
                  <span className="font-medium">{totalJobs} total jobs</span>
                ) : (
                  <>Runner ready</>
                )}
              </>
            ) : (
              <>Runner stopped</>
            )}
          </Link>
        </TooltipTrigger>
        <TooltipContent>
          <p>Runner type: {data.runner_type}</p>
          <p>Status: {data.status}</p>
          <p>Active workers: {activeWorkers}</p>
          <p>Total workers: {data.total_workers || 0}</p>
          <p>Pending jobs: {pendingJobs}</p>
          <p>Initializing jobs: {initializingJobs}</p>
          <p>Running jobs: {runningJobs}</p>
          <p>Total jobs: {data.total_jobs}</p>
          <p className="text-xs mt-1 text-muted-foreground">
            Click for details
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
