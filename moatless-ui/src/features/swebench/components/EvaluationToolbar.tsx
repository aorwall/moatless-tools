import { Button } from "@/lib/components/ui/button";
import { Play, Copy, RefreshCw } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { Evaluation } from "../api/evaluation";

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
  return (
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
                disabled={isStartPending}
              >
                <Play className="h-4 w-4 mr-1" />
                Start
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Start or restart evaluation</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
} 