import { useJobStatusSummary } from "../hooks/useJobStatusSummary";
import { cn } from "@/lib/utils";
import { Loader2, AlertCircle } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";

interface JobStatusBarProps {
  evaluationId: string;
}

export function JobStatusBar({ evaluationId }: JobStatusBarProps) {
  const {
    data: jobStatus,
    isLoading,
    error,
  } = useJobStatusSummary(evaluationId);

  if (isLoading) {
    return (
      <div className="flex items-center text-xs text-muted-foreground">
        <Loader2 className="h-3 w-3 mr-1 animate-spin" />
        Loading job status...
      </div>
    );
  }

  if (error || !jobStatus) {
    return (
      <div className="flex items-center text-xs text-red-500">
        <AlertCircle className="h-3 w-3 mr-1" />
        Error loading job status
      </div>
    );
  }

  const { queued_jobs, running_jobs, total_jobs } = jobStatus;
  const activeJobs = queued_jobs + running_jobs;

  if (activeJobs === 0) {
    return null; // Don't show anything if no active jobs
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={cn(
              "text-xs font-medium px-2 py-1 rounded flex items-center gap-1",
              "text-blue-700 bg-blue-50 border border-blue-200",
            )}
          >
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            {activeJobs} active job{activeJobs !== 1 ? "s" : ""} ({queued_jobs}{" "}
            queued, {running_jobs} running)
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>Total jobs: {total_jobs}</p>
          <p>Queued: {queued_jobs}</p>
          <p>Running: {running_jobs}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
