import {
  Terminal,
  Folder,
  AlertTriangle,
  ChevronDown,
  Info,
  BarChart,
  Cpu,
} from "lucide-react";
import type { Node, ActionStep } from "@/lib/types/trajectory";
import { truncateMessage } from "@/lib/utils/text";
import { cn } from "@/lib/utils";
import { useMemo, Fragment } from "react";
import React from "react";

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

const formatTokenCount = (prompt: number = 0, completion: number = 0, cached: number = 0) => {
  return (
    <div className="flex items-center gap-1 text-[10px] bg-gray-50 px-1.5 py-0.5 rounded">
      <span className="font-mono text-gray-600">{prompt}p</span>
      {cached > 0 && (
        <>
          <span className="font-mono text-gray-400">({cached})</span>
        </>
      )}
      <span className="text-gray-400">+</span>
      <span className="font-mono text-gray-600">{completion}c</span>
    </div>
  );
};

export const TrajectoryNode = ({
  node,
  expanded = false,
}: TrajectoryNodeProps) => {
  const lastAction = node.actionSteps[node.actionSteps.length - 1]?.action.name;
  const showWorkspace = lastAction !== "Finish" && node.fileContext;
  const hasNodeUsage = node.usage && node.usage.prompt_tokens;

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
    // Define priority properties that should be displayed first
    const priorityProps = ['path', 'class_name', 'function_name'];
    
    return (
      <div className="flex flex-col gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-sm text-gray-700">{action.name}</span>
          {priorityProps.map(propKey => 
            action.properties?.[propKey] ? (
              <span className="font-mono text-xs text-gray-500 truncate max-w-[300px]">
                {action.properties[propKey]}
              </span>
            ) : null
          )}
        </div>
        {action.properties && Object.keys(action.properties).length > 0 && (
          <div className="text-xs text-gray-500 pl-4 space-y-0.5">
            {Object.entries(action.properties)
              .filter(([key]) => !priorityProps.includes(key))
              .filter((value) => value != null)
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
  };

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
            <div className="flex items-center justify-between">
              <div className="flex-1">
                {formatActionDisplay(group.action.action)}
              </div>
              
              {/* Node total usage - show only on first action */}
              {index === 0 && hasNodeUsage && (
                <div className="flex items-center gap-1">
                  <Cpu className="h-3 w-3 text-gray-400" />
                  {formatTokenCount(
                    node.usage?.prompt_tokens || 0,
                    node.usage?.completion_tokens || 0,
                    node.usage?.cache_read_tokens || 0
                  )}
                </div>
              )}
            </div>

            {/* Metadata Row */}
            <div className="flex items-center gap-2 mt-1">
              {/* Repeat Counter */}
              {group.count > 1 && (
                <span className="text-[10px] text-gray-400 tabular-nums">
                  ×{group.count}
                </span>
              )}

              {/* Token Usage for this Action Step (if available) */}
              {group.action.completion?.usage && (
                <div className="flex items-center gap-1">
                  {formatTokenCount(
                    group.action.completion.usage.prompt_tokens || 0,
                    group.action.completion.usage.completion_tokens || 0,
                    group.action.completion.usage.cache_read_tokens || 0
                  )}
                </div>
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
