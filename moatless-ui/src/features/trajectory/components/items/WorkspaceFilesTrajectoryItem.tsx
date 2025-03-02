import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip.tsx";
import {
  calculateTotalChanges,
  countPatchChanges,
  truncateFilePath,
} from "@/lib/hooks/useFileUtils.ts";
import {
  FileCode,
  FileEdit,
  FilePlus,
  GitBranch
} from "lucide-react";
import { type FC } from "react";

interface Span {
  span_id: string;
  start_line: number;
  end_line: number;
  tokens?: number;
  pinned?: boolean;
}

interface File {
  file_path: string;
  is_new?: boolean;
  was_edited?: boolean;
  tokens?: number;
  patch?: string;
  spans?: Span[];
  show_all_spans?: boolean;
}

export interface WorkspaceFilesTimelineContent {
  updatedFiles: Array<{
    file_path: string;
    is_new?: boolean;
    has_patch?: boolean;
    patch?: string;
    tokens?: number;
  }>;
  files?: File[];
}

export interface WorkspaceFilesTrajectoryItemProps {
  content: WorkspaceFilesTimelineContent;
}

export const WorkspaceFilesTrajectoryItem: FC<
  WorkspaceFilesTrajectoryItemProps
> = ({ content }) => {
  // Calculate total changes across all files
  const totalChanges = calculateTotalChanges(content.files);

  return (
    <div className="text-xs text-gray-600">
      {content.updatedFiles?.length ? (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <FileEdit size={12} className="text-blue-600" />
              <span>{content.updatedFiles.length} files updated</span>
            </div>

            {(totalChanges.additions > 0 || totalChanges.deletions > 0) && (
              <div className="flex items-center gap-1 text-[10px] border-l border-gray-300 pl-2">
                <GitBranch size={10} className="text-gray-500" />
                <span className="text-green-700">
                  +{totalChanges.additions}
                </span>
                <span>/</span>
                <span className="text-red-700">-{totalChanges.deletions}</span>
              </div>
            )}
          </div>

          {content.updatedFiles.length <= 3 && (
            <div className="flex flex-col gap-0.5 mt-1">
              {content.updatedFiles.map((file, idx) => {
                const { additions, deletions } = countPatchChanges(
                  file.patch || "",
                );

                return (
                  <TooltipProvider key={idx}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div className="flex items-center justify-between text-[10px] text-gray-500">
                          <div className="flex items-center gap-1">
                            {file.is_new ? (
                              <FilePlus size={10} className="text-green-600" />
                            ) : (
                              <FileCode size={10} className="text-blue-600" />
                            )}
                            <span className="font-mono truncate max-w-[150px]">
                              {truncateFilePath(file.file_path)}
                            </span>
                          </div>

                          {file.patch && (additions > 0 || deletions > 0) && (
                            <div className="flex items-center gap-1 ml-2">
                              <span className="text-green-700">
                                +{additions}
                              </span>
                              <span>/</span>
                              <span className="text-red-700">-{deletions}</span>
                            </div>
                          )}
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        <p className="font-mono text-xs">{file.file_path}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        <span>No file changes</span>
      )}
    </div>
  );
};
