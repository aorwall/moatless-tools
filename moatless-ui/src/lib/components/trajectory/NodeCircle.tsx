import { Circle, Loader2, GitBranch, RotateCcw, Split, GitFork, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { Node } from '@/lib/types/trajectory';
import { useRetryNode } from "@/lib/hooks/useRetryNode";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { useNavigate } from "react-router-dom";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/lib/components/ui/tooltip";
import { useState } from "react";
import { RunLoop } from "@/lib/components/loop/RunLoop";

interface NodeCircleProps {
  node: Node;
  isLastNode: boolean;
  isRunning: boolean;
  onClick: () => void;
}

const COLOR_MAPPINGS = {
  blue: {
    border: 'border-blue-500',
    hover: 'hover:border-blue-600 hover:bg-blue-50',
    text: 'text-blue-500',
    hoverText: 'group-hover:text-blue-600',
    bgLight: 'bg-blue-50',
  },
  red: {
    border: 'border-red-500',
    hover: 'hover:border-red-600 hover:bg-red-50',
    text: 'text-red-500',
    hoverText: 'group-hover:text-red-600',
    bgLight: 'bg-red-50',
  },
  yellow: {
    border: 'border-yellow-500',
    hover: 'hover:border-yellow-600 hover:bg-yellow-50',
    text: 'text-yellow-500',
    hoverText: 'group-hover:text-yellow-600',
    bgLight: 'bg-yellow-50',
  },
  green: {
    border: 'border-green-500',
    hover: 'hover:border-green-600 hover:bg-green-50',
    text: 'text-green-500',
    hoverText: 'group-hover:text-green-600',
    bgLight: 'bg-green-50',
  },
  default: {
    border: 'border-gray-300',
    hover: 'hover:border-gray-400 hover:bg-gray-50',
    text: 'text-gray-400',
    hoverText: 'group-hover:text-gray-500',
    bgLight: 'bg-gray-50',
  },
} as const;


function getNodeColor(node: Node, isRunning: boolean): string {
    if (node.nodeId === 0) return "blue";
    if (node.error) return "red";
    if (node.allNodeErrors.length > 0) return "red";
    if (node.allNodeWarnings.length > 0) return "yellow";
    if (node.executed) return "green";
    return "gray";
  }
  
export function NodeCircle({ node, isLastNode, isRunning, onClick }: NodeCircleProps) {
  const nodeColor = getNodeColor(node, isRunning);
  const colors = COLOR_MAPPINGS[nodeColor as keyof typeof COLOR_MAPPINGS] || COLOR_MAPPINGS.default;
  const showSpinner = node.nodeId !== 0 && isRunning && isLastNode;
  const hasChildren = node.children && node.children.length > 0;
  
  const retryNode = useRetryNode();
  const trajectoryId = useTrajectoryStore((state) => state.trajectoryId);
  const navigate = useNavigate();
  const [showRunLoop, setShowRunLoop] = useState(false);

  const handleRetry = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!trajectoryId) return;

    await retryNode.mutateAsync({
      trajectoryId,
      nodeId: node.nodeId,
    });
  };

  const handleFork = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!trajectoryId) return;
    
    const response = await fetch(`/api/trajectories/${trajectoryId}/fork`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        nodeId: node.nodeId
      })
    });
    
    const data = await response.json();
    navigate(`/trajectory/${data.trajectoryId}`);
  };

  const handleBranch = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowRunLoop(true);
  };

  return (
    <div className="relative group/node">
      {/* Invisible hover area that extends upward */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2 w-24 h-16" />

      {/* Action buttons */}
      {node.nodeId !== 0 && (
        <div className={cn(
          "absolute -top-8 left-1/2 transform -translate-x-1/2",
          "flex items-center justify-center gap-1",
          "opacity-0 translate-y-2 pointer-events-none",
          "transition-all duration-200 ease-in-out",
          "group-hover/node:opacity-100 group-hover/node:translate-y-0 group-hover/node:pointer-events-auto",
          "bg-white rounded-full px-2 py-1 shadow-sm",
          "z-30"
        )}>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className="rounded-full p-1 transition-colors duration-200 
                           text-gray-600 hover:text-gray-900"
                  onClick={handleRetry}
                  disabled={retryNode.isPending || !trajectoryId}
                >
                  <RotateCcw className={cn("h-3 w-3", {
                    "animate-spin": retryNode.isPending
                  })} />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Retry this node (recreate and re-execute actions)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className="rounded-full p-1 transition-colors duration-200 
                           text-gray-600 hover:text-gray-900"
                  onClick={handleBranch}
                >
                  <Split className="h-3 w-3" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Expand from this node (create parallel branch)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className="rounded-full p-1 transition-colors duration-200 
                           text-gray-600 hover:text-gray-900"
                  onClick={handleFork}
                  disabled={!trajectoryId}
                >
                  <GitFork className="h-3 w-3" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Fork to new trajectory (create new trajectory from this node)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {hasChildren && (
        <div className="absolute left-0 top-[90%] -translate-x-full -translate-y-1/2 z-30">
          <div 
            className={cn(
              "flex items-center justify-center",
              "bg-white rounded-full h-5 px-1.5",
              "border shadow-sm",
              colors.border,
              "transition-all duration-200",
              "mr-2"
            )}
          >
            <GitBranch className={cn("h-3 w-3 mr-0.5", colors.text)} />
            <span className={cn(
              "text-xs font-medium tabular-nums",
              colors.text
            )}>
              {node.children.length}
            </span>
          </div>
        </div>
      )}

      <button
        className={cn(
          'relative z-20 flex items-center justify-center',
          'h-10 w-10 rounded-full border-2 bg-white',
          'transition-all duration-200 ease-in-out',
          'shadow-sm hover:shadow-md',
          '-ml-5 -mr-5',
          colors.border,
          colors.hover,
          'hover:scale-105'
        )}
        onClick={onClick}
      >
        {showSpinner ? (
          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
        ) : (
          <div className="flex flex-col items-center justify-center">
            {node.nodeId === 0 ? (
              <span className={cn(
                "text-xs font-semibold",
                "text-blue-700"
              )}>
                Start
              </span>
            ) : (
              <span className={cn(
                "text-sm font-medium tabular-nums",
                colors.text
              )}>
                {node.nodeId}
              </span>
            )}
          </div>
        )}
      </button>

      {(node.error || node.allNodeWarnings.length > 0) && (
        <div className="absolute -right-6 top-0 flex flex-col gap-1 z-20">
          {node.error && (
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-red-100 shadow-sm">
              <AlertTriangle className="h-3 w-3 text-red-500" />
            </span>
          )}
          {node.allNodeWarnings.length > 0 && (
            <span className="flex h-4 w-4 items-center justify-center rounded-full bg-yellow-100 shadow-sm">
              <AlertTriangle className="h-3 w-3 text-yellow-500" />
            </span>
          )}
        </div>
      )}

      <RunLoop
        open={showRunLoop}
        onOpenChange={setShowRunLoop}
        defaultMessage={node.userMessage || ""}
        mode="expand"
        trajectoryId={trajectoryId || undefined}
        nodeId={node.nodeId}
      />
    </div>
  );
} 