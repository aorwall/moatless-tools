import {
  MessageSquare,
  Bot,
  Terminal,
  Folder,
  AlertTriangle,
  Split,
  RotateCcw,
  GitFork,
  ChevronDown,
} from "lucide-react";
import type { Node, ActionStep } from "@/lib/types/trajectory";
import { truncateMessage } from "@/lib/utils/text";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { cn } from "@/lib/utils";
import { useMemo, Fragment } from "react";

interface TrajectoryNodeProps {
  node: Node;
  expanded?: boolean;
  level?: number;
}

interface ActionGroup {
  action: ActionStep;
  count: number;
  references: { step: number; count: number }[];
}

// File Updates Section
const FileUpdatesSection = ({ fileContext }: { fileContext: NonNullable<Node['fileContext']> }) => {
  if (!fileContext.updatedFiles?.length) return null;
  
  return (
    <div className="flex items-start gap-1.5 text-xs text-gray-500">
      <Folder className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div className="space-y-1">
        {fileContext.updatedFiles.map((file, index) => (
          <div key={index} className="flex items-center gap-1.5">
            <span className="font-mono">{file.file_path}</span>
            <span className={cn("text-[10px]", {
              "text-blue-600": file.status === "modified",
              "text-green-600": file.status === "added_to_context",
              "text-purple-600": file.status === "updated_context",
            })}>
              {file.status.replace(/_/g, " ")}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Warnings Section
const WarningsSection = ({ fileContext }: { fileContext: NonNullable<Node['fileContext']> }) => {
  if (!fileContext.warnings?.length) return null;
  
  return (
    <div className="flex flex-wrap gap-1.5 text-xs text-yellow-600">
      {fileContext.warnings.map((warning, index) => (
        <div key={index} className="flex items-center gap-1">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
          <span>{warning}</span>
        </div>
      ))}
    </div>
  );
};

export const TrajectoryNode = ({
  node,
  expanded = false,
  level = 0,
}: TrajectoryNodeProps) => {
  const lastAction = node.actionSteps[node.actionSteps.length - 1]?.action.name;
  const showWorkspace = lastAction !== "Finish" && node.fileContext;

  // Group identical actions by their command
  const groupedActions = useMemo(() => {
    return node.actionSteps?.reduce<ActionGroup[]>((acc, step) => {
      const lastGroup = acc[acc.length - 1];
      // Use action.name instead of shortSummary
      if (lastGroup && lastGroup.action.action.name === step.action.name) {
        lastGroup.count++;
        return acc;
      }
      acc.push({ action: step, count: 1, references: [] });
      return acc;
    }, []);
  }, [node.actionSteps]);

  const formatActionDisplay = (action: ActionStep['action']) => {
    // Extract the path from properties if it exists
    const path = action.properties?.path;
    
    // Format based on action type
    switch (action.name) {
      case 'ViewCode':
      case 'StringReplace':
        return (
          <div className="flex flex-col gap-0.5">
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-sm text-gray-700">{action.name}</span>
              {path && (
                <span className="font-mono text-xs text-gray-500 truncate max-w-[300px]">
                  {path}
                </span>
              )}
            </div>
            {expanded && action.properties && Object.keys(action.properties).length > 0 && (
              <div className="text-xs text-gray-500 pl-4 space-y-0.5">
                {Object.entries(action.properties)
                  .filter(([key]) => key !== 'path') // Skip path as it's shown above
                  .map(([key, value]) => (
                    <div key={key} className="flex items-start gap-1">
                      <span className="text-gray-400">{key}:</span>
                      <span className="font-mono truncate max-w-[400px]">
                        {typeof value === 'string' ? value : JSON.stringify(value)}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </div>
        );

      case 'RunTests':
        return (
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-sm text-gray-700">{action.name}</span>
            {action.properties?.no_test_files && (
              <span className="text-[10px] px-1.5 rounded-full bg-yellow-50 text-yellow-700">
                no_test_files
              </span>
            )}
          </div>
        );

      default:
        return (
          <span className="font-mono text-sm text-gray-700">
            {action.name}
          </span>
        );
    }
  };

  if (node.nodeId === 0 && node.userMessage) {
    return (
      <div className="text-xs text-muted-foreground">
        {truncateMessage(node.userMessage)}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {/* Grouped Action Steps */}
      {groupedActions?.map((group, index) => (
        <div 
          key={index} 
          className={cn(
            "group relative flex items-start gap-2 py-1 pl-2 -ml-2 rounded-md transition-colors",
            "hover:bg-gray-50",
            {
              "opacity-90": group.count > 1
            }
          )}
        >
          <Terminal className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-500" />
          <div className="min-w-0 flex-1">
            {/* Action Content */}
            {formatActionDisplay(group.action.action)}

            {/* Metadata Row */}
            <div className="flex items-center gap-2 mt-1">
              {/* Repeat Counter */}
              {group.count > 1 && (
                <span className="text-[10px] text-gray-400 tabular-nums">
                  ×{group.count}
                </span>
              )}

              {/* Status Indicators */}
              {group.action.errors?.length > 0 && (
                <span className="text-xs text-red-600 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  {group.action.errors[0]}
                </span>
              )}

              {/* References */}
              {group.references && group.references.length > 0 && (
                <span className="text-[10px] text-gray-400">
                  Same as step{' '}
                  {group.references.map((ref, i) => (
                    <Fragment key={ref.step}>
                      {i > 0 && ', '}
                      <button className="hover:text-gray-600 underline-offset-2 hover:underline">
                        {ref.step}
                      </button>
                    </Fragment>
                  ))}
                </span>
              )}
            </div>

            {/* Expand Button - only show if has properties */}
            {group.action.action.properties && 
             Object.keys(group.action.action.properties).length > 0 && (
              <button 
                onClick={() => {/* Toggle expanded state */}}
                className="absolute right-2 top-1 opacity-0 group-hover:opacity-100 
                           text-gray-400 hover:text-gray-600 transition-opacity"
              >
                <ChevronDown className={cn(
                  "h-3 w-3 transition-transform",
                  { "transform rotate-180": expanded }
                )} />
              </button>
            )}
          </div>
        </div>
      ))}

      {/* File Context */}
      {showWorkspace && node.fileContext && (
        <>
          {node.fileContext.updatedFiles?.map((file, index) => (
            <div key={index} className="flex items-start gap-1.5 text-xs text-gray-500">
              <Folder className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="font-mono">{file.file_path}</span>
            </div>
          ))}
          {node.fileContext.warnings?.map((warning, index) => (
            <div key={index} className="flex items-center gap-1.5 text-xs text-yellow-600">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              {warning}
            </div>
          ))}
        </>
      )}
    </div>
  );
};
