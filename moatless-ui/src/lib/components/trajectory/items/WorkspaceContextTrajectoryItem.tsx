import { type FC } from "react";
import { Badge } from "@/lib/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { FolderOpen } from "lucide-react";

interface Span {
  span_id: string;
  start_line: number;
  end_line: number;
  tokens?: number;
  pinned?: boolean;
}

interface ContextFile {
  file_path: string;
  tokens?: number;
  spans?: Span[];
}

export interface WorkspaceContextTimelineContent {
  files: ContextFile[];
  max_tokens?: number;
}

export interface WorkspaceContextTrajectoryItemProps {
  content: WorkspaceContextTimelineContent;
  expandedState: boolean;
}

// Helper function to truncate file paths
const truncateFilePath = (filePath: string, maxLength: number = 30): string => {
  if (filePath.length <= maxLength) return filePath;
  
  const parts = filePath.split('/');
  const fileName = parts.pop() || '';
  
  // If just the filename is too long, truncate it
  if (fileName.length >= maxLength - 3) {
    return '...' + fileName.substring(fileName.length - (maxLength - 3));
  }
  
  // Otherwise, keep the filename and add as many directories as possible
  let result = fileName;
  let remainingLength = maxLength - fileName.length - 3; // -3 for the "..."
  
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    // +1 for the slash
    if (part.length + 1 <= remainingLength) {
      result = part + '/' + result;
      remainingLength -= (part.length + 1);
    } else {
      break;
    }
  }
  
  return '...' + (result.startsWith('/') ? result : '/' + result);
};

export const WorkspaceContextTrajectoryItem: FC<WorkspaceContextTrajectoryItemProps> = ({
  content,
  expandedState,
}) => {
  if (!expandedState) {
    return (
      <div className="text-xs text-gray-600">
        {content.files?.length ? (
          <div className="flex flex-col gap-1">
            <span>{content.files.length} files in context</span>
            {content.files.length <= 3 && (
              <div className="flex flex-col gap-0.5 mt-1">
                {content.files.map((file, idx) => (
                  <TooltipProvider key={idx}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex items-center gap-1 text-[10px] text-gray-500">
                          <FolderOpen size={10} />
                          <span className="font-mono truncate max-w-[200px]">
                            {truncateFilePath(file.file_path)}
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        <p className="font-mono text-xs">{file.file_path}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ))}
              </div>
            )}
          </div>
        ) : (
          <span>No files in context</span>
        )}
      </div>
    );
  }

  // Calculate total tokens used
  const totalTokens = content.files.reduce((sum, file) => sum + (file.tokens || 0), 0);
  const maxTokens = content.max_tokens || 0;
  const tokenPercentage = maxTokens ? Math.min(100, Math.round((totalTokens / maxTokens) * 100)) : 0;

  return (
    <div className="space-y-4">
      {/* Context Files Section */}
      {content.files && content.files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-gray-700">
              Files in Context
            </div>
            <div className="text-xs text-gray-500">
              {content.files.length} files
            </div>
          </div>

          {/* Token usage */}
          {maxTokens > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-600">Token Usage</span>
                <span className="text-gray-600">{totalTokens} / {maxTokens} ({tokenPercentage}%)</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
                <div 
                  className="h-full rounded-full bg-blue-500" 
                  style={{ width: `${tokenPercentage}%` }}
                />
              </div>
            </div>
          )}

          <div className="space-y-2 rounded-md bg-gray-50 p-3">
            {content.files.map((file, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex items-center gap-2 max-w-[70%]">
                          <FolderOpen size={14} className="text-gray-500 flex-shrink-0" />
                          <span className="font-mono text-gray-700 truncate">
                            {truncateFilePath(file.file_path, 40)}
                          </span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="top">
                        <p className="font-mono text-xs">{file.file_path}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  <div className="flex items-center gap-2">
                    {file.spans && (
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0.5">
                        {file.spans.length} spans
                      </Badge>
                    )}
                    <Badge variant="outline" className="text-[10px] px-1.5 py-0.5">
                      {file.tokens} tokens
                    </Badge>
                  </div>
                </div>

                {file.spans && file.spans.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 rounded border border-gray-200 p-2">
                    {file.spans.map((span) => (
                      <div
                        key={span.span_id}
                        className={cn(
                          "inline-flex items-center gap-1 rounded border border-gray-200 px-2 py-0.5 text-[10px] text-gray-700",
                          span.pinned && "bg-purple-50 border-purple-200"
                        )}
                      >
                        <span>{span.span_id}</span>
                        <span className="text-gray-400">
                          ({span.start_line}-{span.end_line})
                        </span>
                        {span.pinned && (
                          <span className="rounded bg-purple-50 px-1 text-purple-600">
                            📌
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}; 