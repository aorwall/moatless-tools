import { useRunnerInfo } from '../hooks/useRunnerInfo';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Loader2, AlertCircle, Activity, Server } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/lib/components/ui/tooltip';
import { RunnerStatus } from '../types';

export function RunnerStatusBar() {
  const { data, isLoading, error } = useRunnerInfo();

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
            <Link to="/runner" className="flex items-center text-xs text-red-500">
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

  const { info, jobs } = data;
  const runnerIsUp = info.status === RunnerStatus.RUNNING;
  const activeWorkers = info.data.active_workers || 0;
  
  // Count jobs by status
  const queuedJobs = jobs.filter(job => job.status === 'queued').length;
  const runningJobs = jobs.filter(job => job.status === 'running').length;
  const activeJobs = queuedJobs + runningJobs;
  
  if (!runnerIsUp && activeJobs === 0) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Link to="/runner" className="flex items-center text-xs text-orange-500">
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
                ? (activeJobs > 0 
                    ? "text-blue-700 bg-blue-50 border border-blue-200" 
                    : "text-green-700 bg-green-50 border border-green-200")
                : "text-orange-700 bg-orange-50 border border-orange-200"
            )}
          >
            <span className={cn(
              "w-2 h-2 rounded-full",
              runnerIsUp 
                ? (activeJobs > 0 ? "bg-blue-500 animate-pulse" : "bg-green-500") 
                : "bg-orange-500"
            )}></span>
            {runnerIsUp ? (
              <>
                {activeJobs > 0 ? (
                  <>
                    {activeJobs} job{activeJobs !== 1 ? 's' : ''} active
                  </>
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
          <p>Runner type: {info.runner_type}</p>
          <p>Status: {info.status}</p>
          <p>Active workers: {activeWorkers}</p>
          <p>Total workers: {info.data.total_workers || 0}</p>
          <p>Queued jobs: {queuedJobs}</p>
          <p>Running jobs: {runningJobs}</p>
          <p className="text-xs mt-1 text-muted-foreground">Click for details</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
} 