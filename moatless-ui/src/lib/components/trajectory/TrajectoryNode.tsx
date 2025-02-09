import {
  MessageSquare,
  Bot,
  Terminal,
  Folder,
  AlertTriangle,
  Split,
  RotateCcw,
  Git,
  GitFork,
} from "lucide-react";
import type { Node } from "@/lib/types/trajectory";
import { truncateMessage } from "@/lib/utils/text";
import { Button } from "@/lib/components/ui/button";
import { RunLoop } from "@/lib/components/loop/RunLoop";
import { useState } from "react";
import { useRetryNode } from "@/lib/hooks/useRetryNode";
import { cn } from "@/lib/utils";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/lib/components/ui/tooltip";
import { useNavigate } from "react-router-dom";

interface TrajectoryNodeProps {
  node: Node;
  expanded?: boolean;
}

export const TrajectoryNode = ({
  node,
  expanded = false,
}: TrajectoryNodeProps) => {
  const [showRunLoop, setShowRunLoop] = useState(false);
  const retryNode = useRetryNode();
  const lastAction = node.actionSteps[node.actionSteps.length - 1]?.action.name;
  const showWorkspace = lastAction !== "Finish" && node.fileContext;
  const trajectoryId = useTrajectoryStore((state) => state.trajectoryId);
  const navigate = useNavigate();

  const handleRetry = async () => {
    if (!trajectoryId) return;

    await retryNode.mutateAsync({
      trajectoryId,
      nodeId: node.nodeId,
    });
  };

  const handleFork = async () => {
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

  if (node.nodeId === 0 && node.userMessage) {
    return (
      <div className="break-words text-left text-xs text-muted-foreground">
        {truncateMessage(node.userMessage)}
      </div>
    );
  }

  return (
    <div className="flex gap-4">
      <div className="flex-1">
        {node.error && (
          <div className="mb-2 flex items-start gap-2 text-left">
            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-destructive/70" />
            <div className="min-w-0 flex-1">
              <span className="break-words text-xs text-destructive/70">
                Error occurred
              </span>
            </div>
          </div>
        )}

        {(node.userMessage || node.assistantMessage) && (
          <div className="flex flex-col gap-1 text-left text-muted-foreground">
            {node.userMessage && (
              <div className="flex items-start gap-2">
                <MessageSquare className="mt-0.5 h-3 w-3 shrink-0" />
                <div className="min-w-0 flex-1">
                  <span className="break-words text-xs">User</span>
                </div>
              </div>
            )}
            {node.assistantMessage && (
              <div className="flex items-start gap-2">
                <Bot className="mt-0.5 h-3 w-3 shrink-0" />
                <div className="min-w-0 flex-1">
                  <span className="break-words text-xs">Assistant</span>
                </div>
              </div>
            )}
          </div>
        )}

        {node.actionSteps?.length > 0 && (
          <div className="space-y-1.5 text-left">
            {node.actionSteps.map((step, index) => (
              <div key={index} className="flex items-start gap-2">
                <Terminal className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="break-all font-mono text-xs">
                      {step.action.shortSummary}
                    </span>
                    {step.errors?.map((error, i) => (
                      <span
                        key={i}
                        className="break-words rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] text-destructive/70"
                      >
                        {error}
                      </span>
                    ))}
                    {step.warnings?.map((warning, i) => (
                      <span
                        key={i}
                        className="bg-warning/10 text-warning/70 break-words rounded px-1.5 py-0.5 text-[10px]"
                      >
                        {warning}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {showWorkspace && (
          <div className="space-y-2 text-left">
            {node.fileContext?.warnings?.length && node.fileContext?.warnings?.length > 0 && (
              <div className="flex items-start gap-2">
                <AlertTriangle className="text-warning/70 mt-0.5 h-3 w-3 shrink-0" />
                <div className="flex flex-wrap gap-1">
                  {node.fileContext?.warnings?.map((warning, index) => (
                    <span
                      key={index}
                      className="bg-warning/10 text-warning/70 break-words rounded px-1.5 py-0.5 text-[10px]"
                    >
                      {warning}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {node.fileContext?.updatedFiles?.length ? (
              <div className="flex items-start gap-2">
                <Folder className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                <div className="space-y-1">
                  {node.fileContext.updatedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex flex-wrap items-center gap-1 text-[10px] text-muted-foreground"
                    >
                      <span className="break-all font-mono">
                        {file.file_path}
                      </span>
                      {file.status === "modified" && (
                        <span className="rounded bg-primary/10 px-1.5 py-0.5 text-primary/70">
                          modified
                        </span>
                      )}
                      {file.status === "added_to_context" && (
                        <span className="rounded bg-secondary/10 px-1.5 py-0.5 text-secondary/70">
                          added to context
                        </span>
                      )}
                      {file.status === "updated_context" && (
                        <span className="rounded bg-accent/10 px-1.5 py-0.5 text-accent/70">
                          updated context
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ) : node.nodeId !== 0 ? (
              <div className="flex items-start gap-2">
                <Folder className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                <span className="rounded bg-destructive/10 px-1.5 py-0.5 text-[10px] text-destructive/70">
                  No updates to workspace
                </span>
              </div>
            ) : null}
          </div>
        )}
      </div>
      {node.nodeId !== 0 && (
        <div className="flex-none flex gap-1">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={handleRetry}
                  disabled={retryNode.isPending || !trajectoryId}
                >
                  <RotateCcw className={cn("h-4 w-4", {
                    "animate-spin": retryNode.isPending
                  })} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Retry this node (recreate and re-execute actions)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => setShowRunLoop(true)}
                >
                  <Split className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Expand from this node (create parallel branch)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={handleFork}
                  disabled={!trajectoryId}
                >
                  <GitFork className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Fork to new trajectory (create new trajectory from this node)</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
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
};
