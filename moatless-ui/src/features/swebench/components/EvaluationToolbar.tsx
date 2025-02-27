import { Button } from "@/lib/components/ui/button";
import { Play, Copy, RefreshCw, AlertCircle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { Evaluation } from "../api/evaluation";
import { useRunnerStatus } from "@/lib/hooks/useRunnerStatus";
import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";

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
  canStart 
}: EvaluationToolbarProps) {
  const { data: runnerStatus, isLoading: isRunnerStatusLoading } = useRunnerStatus();
  
  const activeWorkers = runnerStatus?.info?.data?.active_workers ?? 0;
  const totalWorkers = runnerStatus?.info?.data?.total_workers ?? 0;
  const runnerStatusText = runnerStatus?.info?.status ?? 'unknown';
  const hasActiveWorkers = activeWorkers > 0;
  const workersRunning = !isRunnerStatusLoading && hasActiveWorkers;
  const startDisabled = isStartPending || !canStart || !workersRunning;
  
  return (
    <div className="flex flex-col gap-3">
      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" onClick={onClone} disabled={isClonePending}>
                <Copy className="h-4 w-4 mr-1" />
                Clone
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Clone this evaluation</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        
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
        
        {canStart && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button 
                  size="sm" 
                  onClick={onStart} 
                  disabled={startDisabled}
                >
                  {!workersRunning && <AlertCircle className="h-4 w-4 mr-1" />}
                  {workersRunning && <Play className="h-4 w-4 mr-1" />}
                  Start
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {!workersRunning ? (
                  <p>Workers not running. Start workers to enable evaluation.</p>
                ) : (
                  <p>Start or restart evaluation</p>
                )}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      
      {/* Status information */}
      {!workersRunning && (
        <div className="flex items-center">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className={cn(
                  "text-xs font-medium px-2 py-1 rounded flex items-center gap-1",
                  "text-amber-700 bg-amber-50 border border-amber-200"
                )}>
                  <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                  Runner service is not active. Start workers to enable evaluations.
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Status: {runnerStatusText}</p>
                <p>No active workers available. Start the runner service to enable evaluations.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
} 