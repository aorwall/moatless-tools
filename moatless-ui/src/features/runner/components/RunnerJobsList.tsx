import { useState, useMemo } from "react";
import { JobInfo, JobStatus } from "../types";
import { useCancelJob, useRetryJob } from "../hooks/useJobsManagement";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/lib/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { Badge } from "@/lib/components/ui/badge";
import { Button } from "@/lib/components/ui/button";
import {
  Play,
  XCircle,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircleIcon,
} from "lucide-react";
import { dateTimeFormat } from "@/lib/utils/date";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip";

interface RunnerJobsListProps {
  jobs: JobInfo[];
  isLoading?: boolean;
}

export function RunnerJobsList({
  jobs,
  isLoading = false,
}: RunnerJobsListProps) {
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const cancelJob = useCancelJob();
  const retryJob = useRetryJob();

  // Filter jobs based on selected status
  const filteredJobs = useMemo(() => {
    if (statusFilter === "all") {
      return jobs;
    }
    return jobs.filter((job) => job.status === statusFilter);
  }, [jobs, statusFilter]);

  // Parse project and trajectory IDs from job ID
  const parseJobId = (id: string) => {
    // Format: run_projectId_trajectoryId
    const parts = id.split("_");
    if (parts.length >= 3) {
      return {
        projectId: parts[1],
        trajectoryId: parts.slice(2).join("_"), // In case trajectory_id contains underscores
      };
    }
    return { projectId: "", trajectoryId: "" };
  };

  // Get status badge for a job
  const getStatusBadge = (status: JobStatus) => {
    switch (status) {
      case JobStatus.RUNNING:
        return (
          <Badge
            variant="outline"
            className="bg-blue-50 text-blue-700 border-blue-200"
          >
            Running
          </Badge>
        );
      case JobStatus.QUEUED:
        return (
          <Badge
            variant="outline"
            className="bg-yellow-50 text-yellow-700 border-yellow-200"
          >
            Queued
          </Badge>
        );
      case JobStatus.COMPLETED:
        return (
          <Badge
            variant="outline"
            className="bg-green-50 text-green-700 border-green-200"
          >
            Completed
          </Badge>
        );
      case JobStatus.FAILED:
        return (
          <Badge
            variant="outline"
            className="bg-red-50 text-red-700 border-red-200"
          >
            Failed
          </Badge>
        );
      case JobStatus.CANCELED:
        return (
          <Badge
            variant="outline"
            className="bg-orange-50 text-orange-700 border-orange-200"
          >
            Canceled
          </Badge>
        );
      default:
        return (
          <Badge
            variant="outline"
            className="bg-gray-50 text-gray-700 border-gray-200"
          >
            Pending
          </Badge>
        );
    }
  };

  // Handle cancel button click
  const handleCancel = (jobId: string) => {
    const { projectId, trajectoryId } = parseJobId(jobId);
    if (projectId && trajectoryId) {
      cancelJob.mutate({ projectId, trajectoryId });
    }
  };

  // Handle retry button click
  const handleRetry = (jobId: string) => {
    const { projectId, trajectoryId } = parseJobId(jobId);
    if (projectId && trajectoryId) {
      retryJob.mutate({ projectId, trajectoryId });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Jobs</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Filter:</span>
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Jobs</SelectItem>
              <SelectItem value="queued">Queued</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="failed">Failed</SelectItem>
              <SelectItem value="canceled">Canceled</SelectItem>
              <SelectItem value="pending">Pending</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="border rounded-md">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Job ID</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Queued At</TableHead>
              <TableHead>Started At</TableHead>
              <TableHead>Ended At</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredJobs.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={6}
                  className="text-center py-6 text-muted-foreground"
                >
                  {isLoading ? "Loading..." : "No jobs found"}
                </TableCell>
              </TableRow>
            ) : (
              filteredJobs.map((job) => {
                const { projectId, trajectoryId } = parseJobId(job.id);
                return (
                  <TableRow key={job.id}>
                    <TableCell className="font-mono text-xs">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="cursor-help">{job.id}</span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>Project: {projectId}</p>
                            <p>Trajectory: {trajectoryId}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                    <TableCell>{getStatusBadge(job.status)}</TableCell>
                    <TableCell className="text-sm">
                      {job.enqueued_at
                        ? dateTimeFormat.format(new Date(job.enqueued_at))
                        : "-"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {job.started_at
                        ? dateTimeFormat.format(new Date(job.started_at))
                        : "-"}
                    </TableCell>
                    <TableCell className="text-sm">
                      {job.ended_at
                        ? dateTimeFormat.format(new Date(job.ended_at))
                        : "-"}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        {(job.status === JobStatus.QUEUED ||
                          job.status === JobStatus.RUNNING) && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleCancel(job.id)}
                            disabled={cancelJob.isPending}
                          >
                            <XCircle className="h-4 w-4 text-red-500" />
                            <span className="sr-only">Cancel</span>
                          </Button>
                        )}

                        {job.status === JobStatus.FAILED && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleRetry(job.id)}
                            disabled={retryJob.isPending}
                          >
                            <Play className="h-4 w-4 text-green-500" />
                            <span className="sr-only">Retry</span>
                          </Button>
                        )}

                        {job.status === JobStatus.FAILED && job.exc_info && (
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <AlertCircle className="h-4 w-4 text-amber-500" />
                                  <span className="sr-only">View Error</span>
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-md">
                                <p className="font-bold">Error:</p>
                                <pre className="text-xs mt-1 max-h-40 overflow-auto p-2 bg-muted rounded">
                                  {job.exc_info}
                                </pre>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
