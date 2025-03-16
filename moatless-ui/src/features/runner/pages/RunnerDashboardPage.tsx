import { useRunnerInfo } from "../hooks/useRunnerInfo";
import { RunnerJobsList } from "../components/RunnerJobsList";
import { useCancelJob } from "../hooks/useJobsManagement";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/lib/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/lib/components/ui/tabs";
import { Button } from "@/lib/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { Separator } from "@/lib/components/ui/separator";
import { Badge } from "@/lib/components/ui/badge";
import { Input } from "@/lib/components/ui/input";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Server,
  Settings,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import { RunnerStatus } from "../types";
import { toast } from "sonner";

export function RunnerDashboardPage() {
  const { data, isLoading, error, refetch } = useRunnerInfo();
  const cancelJob = useCancelJob();
  const [projectIdToCancel, setProjectIdToCancel] = useState("");

  if (isLoading) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Runner Dashboard</h1>
        <div className="grid grid-cols-4 gap-4">
          {Array(4)
            .fill(0)
            .map((_, i) => (
              <Card key={i}>
                <CardHeader className="pb-2">
                  <Skeleton className="h-4 w-1/2" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-8 w-16" />
                </CardContent>
              </Card>
            ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Runner Dashboard</h1>
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            Failed to load runner information.{" "}
            {error instanceof Error ? error.message : "Unknown error"}
            <div className="mt-2">
              <Button variant="outline" size="sm" onClick={() => refetch()}>
                Retry
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const { info, jobs } = data;
  const runnerIsUp = info.status === RunnerStatus.RUNNING;
  const activeWorkers = info.data.active_workers || 0;
  const totalWorkers = info.data.total_workers || 0;

  // Count jobs by status
  const queuedJobs = jobs.filter((job) => job.status === "queued").length;
  const runningJobs = jobs.filter((job) => job.status === "running").length;
  const completedJobs = jobs.filter((job) => job.status === "completed").length;
  const failedJobs = jobs.filter((job) => job.status === "failed").length;
  const canceledJobs = jobs.filter((job) => job.status === "canceled").length;
  const pendingJobs = jobs.filter((job) => job.status === "pending").length;
  const activeJobs = queuedJobs + runningJobs;

  // Extract unique project IDs from job IDs
  const projectIds = Array.from(
    new Set(
      jobs
        .map((job) => {
          const parts = job.id.split("_");
          return parts.length >= 2 ? parts[1] : null;
        })
        .filter(Boolean), // Remove null values
    ),
  );

  // Handle cancel all jobs for a project
  const handleCancelProject = () => {
    if (!projectIdToCancel) {
      toast.error("Please enter a project ID");
      return;
    }

    cancelJob.mutate(
      { projectId: projectIdToCancel },
      {
        onSuccess: () => {
          setProjectIdToCancel("");
        },
      },
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Runner Dashboard</h1>
        <Button onClick={() => refetch()} variant="outline" size="sm">
          Refresh
        </Button>
      </div>

      {/* Runner Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Runner Status
          </CardTitle>
          <CardDescription>
            Current status of the job runner and workers
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground">Status</span>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className={`w-2 h-2 rounded-full ${runnerIsUp ? "bg-green-500" : "bg-red-500"}`}
                ></span>
                <span className="text-lg font-semibold">
                  {runnerIsUp ? "Running" : "Stopped"}
                </span>
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground">Type</span>
              <span className="text-lg font-semibold mt-1">
                {info.runner_type}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground">
                Active Workers
              </span>
              <span className="text-lg font-semibold mt-1">
                {activeWorkers}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground">
                Total Workers
              </span>
              <span className="text-lg font-semibold mt-1">{totalWorkers}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Job Statistics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <Clock className="h-4 w-4 text-blue-500" />
              Queued
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{queuedJobs}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <Activity className="h-4 w-4 text-indigo-500" />
              Running
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{runningJobs}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              Completed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{completedJobs}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              Failed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{failedJobs}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <XCircle className="h-4 w-4 text-orange-500" />
              Canceled
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{canceledJobs}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-1">
              <Settings className="h-4 w-4 text-gray-500" />
              Pending
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{pendingJobs}</p>
          </CardContent>
        </Card>
      </div>

      {/* Projects */}
      {projectIds.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Projects with Jobs</CardTitle>
            <CardDescription>
              Projects that have active or completed jobs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {projectIds.map((projectId) => (
                <Badge key={projectId} variant="outline" className="text-sm">
                  {projectId}
                </Badge>
              ))}
            </div>
            <Separator className="my-4" />
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-medium">
                Cancel all jobs for project
              </h3>
              <div className="flex gap-2">
                <Input
                  type="text"
                  placeholder="Enter project ID"
                  value={projectIdToCancel}
                  onChange={(e) => setProjectIdToCancel(e.target.value)}
                />
                <Button
                  variant="destructive"
                  onClick={handleCancelProject}
                  disabled={cancelJob.isPending || !projectIdToCancel}
                >
                  Cancel Jobs
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Jobs List */}
      <Tabs defaultValue="active">
        <TabsList>
          <TabsTrigger value="active">Active Jobs ({activeJobs})</TabsTrigger>
          <TabsTrigger value="all">All Jobs ({jobs.length})</TabsTrigger>
        </TabsList>
        <TabsContent value="active" className="mt-4">
          <RunnerJobsList
            jobs={jobs.filter(
              (job) => job.status === "queued" || job.status === "running",
            )}
          />
        </TabsContent>
        <TabsContent value="all" className="mt-4">
          <RunnerJobsList jobs={jobs} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
