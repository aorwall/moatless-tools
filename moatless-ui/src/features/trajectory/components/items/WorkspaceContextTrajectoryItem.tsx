import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip.tsx";
import { FolderOpen } from "lucide-react";
import { type FC } from "react";

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
}

// Helper function to truncate file paths
const truncateFilePath = (filePath: string, maxLength: number = 30): string => {
  if (filePath.length <= maxLength) return filePath;

  const parts = filePath.split("/");
  const fileName = parts.pop() || "";

  // If just the filename is too long, truncate it
  if (fileName.length >= maxLength - 3) {
    return "..." + fileName.substring(fileName.length - (maxLength - 3));
  }

  // Otherwise, keep the filename and add as many directories as possible
  let result = fileName;
  let remainingLength = maxLength - fileName.length - 3; // -3 for the "..."

  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    // +1 for the slash
    if (part.length + 1 <= remainingLength) {
      result = part + "/" + result;
      remainingLength -= part.length + 1;
    } else {
      break;
    }
  }

  return "..." + (result.startsWith("/") ? result : "/" + result);
};

export const WorkspaceContextTrajectoryItem: FC<
  WorkspaceContextTrajectoryItemProps
> = ({ content }) => {
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
};
