import {
  Loader2,
  GitBranch,
  RotateCcw,
  Split,
  GitFork,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils.ts";
import type { Node, Trajectory, Reward } from "@/lib/types/trajectory.ts";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/lib/components/ui/tooltip.tsx";
import { useState } from "react";
import { RunLoop } from "@/lib/components/loop/RunLoop.tsx";
import { useNodeActions } from "@/lib/hooks/useNodeActions.ts";

interface NodeCircleProps {
  node: Node;
  isLastNode: boolean;
  isRunning: boolean;
  onClick?: () => void;
  trajectory: Trajectory;
}

// Single color theme for all nodes
const NODE_COLOR = {
  border: "border-slate-400",
  hover: "hover:border-slate-500 hover:bg-slate-50",
  text: "text-slate-600",
  hoverText: "group-hover:text-slate-700",
  bgLight: "bg-slate-50",
} as const;

const NODE_LAYOUT = {
  size: {
    circle: "h-10 w-10",
    actionButton: "h-3 w-3",
    indicator: "h-4 w-4",
  },
  spacing: {
    actionButtonsOffset: "-top-8",
    indicatorOffset: "-right-6",
  },
} as const;

// Function to determine reward badge color
function getRewardColor(reward: Reward) {
  // Clamp the reward value between -100 and 100
  const clampedReward = Math.max(-100, Math.min(100, reward.value));

  if (clampedReward < -50) {
    return "bg-red-50 border-red-200 text-red-700";
  } else if (clampedReward < 0) {
    return "bg-yellow-50 border-yellow-200 text-yellow-700";
  } else if (clampedReward > 50) {
    return "bg-green-50 border-green-200 text-green-700";
  } else {
    return "bg-yellow-50 border-yellow-200 text-yellow-700";
  }
}

function formatReward(reward: number): string {
  return Math.round(reward).toString();
}

export function NodeCircle({
  node,
  isLastNode,
  isRunning,
  onClick,
  trajectory,
}: NodeCircleProps) {
  const colors = NODE_COLOR;
  const showSpinner = node.nodeId !== 0 && isRunning && isLastNode;
  const isBranched = node.children && node.children.length > 1;

  const [showRunLoop, setShowRunLoop] = useState(false);
  const { handleRetry, handleFork, isRetryPending, canPerformActions } =
    useNodeActions({ nodeId: node.nodeId, trajectory });

  const handleRetryClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await handleRetry();
  };

  const handleForkClick = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await handleFork();
  };

  const handleBranch = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowRunLoop(true);
  };

  return (
    <div className="relative group/node">
      {/* Invisible hover area that extends upward */}
      <div className="absolute -top-10 left-1/2 -translate-x-1/2 w-24 h-8" />

      {/* Action buttons container */}
      {node.nodeId !== 0 && (
        <div
          className={cn(
            "absolute left-1/2 transform -translate-x-1/2",
            NODE_LAYOUT.spacing.actionButtonsOffset,
            "flex items-center justify-center gap-1",
            "opacity-0 translate-y-2 pointer-events-none",
            "group-hover/node:opacity-100 group-hover/node:translate-y-0 group-hover/node:pointer-events-auto",
            "bg-white rounded-full px-2 py-1 shadow-sm z-40",
          )}
        >
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  className={cn(
                    "rounded-full p-1 transition-colors duration-200",
                    "text-gray-600 hover:text-gray-900",
                    "z-50",
                  )}
                  onClick={handleRetryClick}
                  disabled={isRetryPending || !canPerformActions}
                >
                  <RotateCcw
                    className={cn("h-3 w-3", {
                      "animate-spin": isRetryPending,
                    })}
                  />
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
                  disabled={!canPerformActions}
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
                  onClick={handleForkClick}
                  disabled={!canPerformActions}
                >
                  <GitFork className="h-3 w-3" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p>
                  Fork to new trajectory (create new trajectory from this node)
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {/* Reward Badge */}
      {node.reward && (
        <div className="absolute -left-6 -top-1 z-30">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    "flex items-center justify-center rounded-full shadow-sm px-1.5 py-0.5",
                    "text-[10px] font-medium",
                    "border",
                    getRewardColor(node.reward),
                  )}
                >
                  {formatReward(node.reward.value)}
                </span>
              </TooltipTrigger>
              <TooltipContent>
                {node.reward.explanation && <p>{node.reward.explanation}</p>}
                {!node.reward.explanation && (
                  <p>No reward explanation available</p>
                )}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}

      {isBranched && (
        <div className="absolute left-1/2 top-full -translate-x-1/2 z-30">
          <div
            className={cn(
              "flex items-center justify-center",
              "bg-white rounded-full h-5 px-1.5",
              "border shadow-sm",
              colors.border,
              "transition-all duration-200",
              "mt-1",
            )}
          >
            <GitBranch className={cn("h-3 w-3 mr-0.5", colors.text)} />
            <span
              className={cn("text-xs font-medium tabular-nums", colors.text)}
            >
              {node.children.length}
            </span>
          </div>
        </div>
      )}

      {/* Main circle button */}
      <button
        className={cn(
          "relative z-20 flex items-center justify-center",
          NODE_LAYOUT.size.circle,
          "rounded-full border-2 bg-white",
          "transition-all duration-200 ease-in-out",
          "shadow-sm hover:shadow-md -ml-5 -mr-5",
          colors.border,
          colors.hover,
          "hover:scale-105",
        )}
        onClick={onClick}
      >
        {showSpinner ? (
          <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
        ) : (
          <div className="flex flex-col items-center justify-center">
            {node.nodeId === 0 ? (
              <span className={cn("text-xs font-semibold", "text-slate-700")}>
                Start
              </span>
            ) : (
              <span
                className={cn("text-sm font-medium tabular-nums", colors.text)}
              >
                {node.nodeId}
              </span>
            )}
          </div>
        )}
      </button>

      {/* Status indicators */}
      <div
        className={cn(
          "absolute flex flex-col gap-1 z-20",
          NODE_LAYOUT.spacing.indicatorOffset,
          "top-0",
        )}
      >
        {node.error && (
          <span
            className={cn(
              "flex items-center justify-center rounded-full bg-red-100 shadow-sm",
              NODE_LAYOUT.size.indicator,
            )}
          >
            <AlertTriangle className={NODE_LAYOUT.size.actionButton} />
          </span>
        )}
        {node.allNodeWarnings.length > 0 && (
          <span
            className={cn(
              "flex items-center justify-center rounded-full bg-yellow-100 shadow-sm",
              NODE_LAYOUT.size.indicator,
            )}
          >
            <AlertTriangle className={NODE_LAYOUT.size.actionButton} />
          </span>
        )}
      </div>

      <RunLoop
        open={showRunLoop}
        onOpenChange={setShowRunLoop}
        defaultMessage={node.userMessage || ""}
        mode="expand"
        trajectoryId={trajectory.id}
        nodeId={node.nodeId}
      />
    </div>
  );
}
