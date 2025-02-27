import { type FC } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip";
import { FileCode, FilePlus, FileEdit, FolderOpen, GitBranch } from "lucide-react";
import { countPatchChanges, truncateFilePath, calculateTotalChanges } from "@/lib/hooks/useFileUtils";

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
  expandedState: boolean;
}

export const WorkspaceFilesTrajectoryItem: FC<WorkspaceFilesTrajectoryItemProps> = ({
  content,
  expandedState,
}) => {
  // Calculate total changes across all files
  const totalChanges = calculateTotalChanges(content.files);

  if (!expandedState) {
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
                  <span className="text-green-700">+{totalChanges.additions}</span>
                  <span>/</span>
                  <span className="text-red-700">-{totalChanges.deletions}</span>
                </div>
              )}
            </div>
            
            {content.updatedFiles.length <= 3 && (
              <div className="flex flex-col gap-0.5 mt-1">
                {content.updatedFiles.map((file, idx) => {
                  const { additions, deletions } = countPatchChanges(file.patch || '');
                  
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
                                <span className="text-green-700">+{additions}</span>
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
  }

  // Count new vs modified files
  const newFiles = content.updatedFiles?.filter(f => f.is_new)?.length || 0;
  const modifiedFiles = content.updatedFiles?.filter(f => !f.is_new && f.patch)?.length || 0;

  return (
    <div className="space-y-4">
      {/* Updated Files Section */}
      {content.updatedFiles && content.updatedFiles.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-gray-700">
              Updated Files
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              {newFiles > 0 && (
                <div className="flex items-center gap-1">
                  <FilePlus size={12} className="text-green-600" />
                  <span className="text-green-700">{newFiles} new</span>
                </div>
              )}
              {modifiedFiles > 0 && (
                <div className="flex items-center gap-1">
                  <FileEdit size={12} className="text-blue-600" />
                  <span className="text-blue-700">{modifiedFiles} modified</span>
                </div>
              )}
              {(totalChanges.additions > 0 || totalChanges.deletions > 0) && (
                <div className="flex items-center gap-1 border-l border-gray-300 pl-2">
                  <GitBranch size={12} className="text-gray-500" />
                  <span className="text-green-700">+{totalChanges.additions}</span>
                  <span>/</span>
                  <span className="text-red-700">-{totalChanges.deletions}</span>
                </div>
              )}
            </div>
          </div>
          <div className="space-y-2 rounded-md bg-gray-50 p-3">
            {content.updatedFiles.map((file, idx) => {
              const { additions, deletions } = countPatchChanges(file.patch || '');
              
              return (
                <div key={idx} className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex items-center gap-2 max-w-[70%]">
                            {file.is_new ? (
                              <FilePlus size={14} className="text-green-600 flex-shrink-0" />
                            ) : (
                              <FileEdit size={14} className="text-blue-600 flex-shrink-0" />
                            )}
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
                      {file.is_new && (
                        <Badge variant="default" className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-800">
                          new
                        </Badge>
                      )}
                      {file.has_patch && (
                        <Badge variant="default" className="text-[10px] px-1.5 py-0.5 bg-blue-100 text-blue-800">
                          modified
                        </Badge>
                      )}
                      {file.tokens && (
                        <Badge variant="outline" className="text-[10px] px-1.5 py-0.5">
                          {file.tokens} tokens
                        </Badge>
                      )}
                      {file.patch && (additions > 0 || deletions > 0) && (
                        <div className="flex items-center gap-1 text-[10px]">
                          <span className="text-green-700">+{additions}</span>
                          <span>/</span>
                          <span className="text-red-700">-{deletions}</span>
                        </div>
                      )}
                    </div>
                  </div>


                  {file.patch && (
                    <div className="max-w-full rounded bg-gray-100 p-2">
                      <pre className="overflow-x-auto whitespace-pre font-mono text-[10px]">
                        {file.patch?.split("\n")
                          .map((line, i) => (
                            <div
                              key={i}
                              className={cn(
                                "leading-5 px-1",
                                line.startsWith("+") && !line.startsWith("+++") && "bg-green-50 text-green-700",
                                line.startsWith("-") && !line.startsWith("---") && "bg-red-50 text-red-700",
                                line.startsWith("@@") && "text-purple-600 border-t border-b border-gray-200 bg-gray-100 py-0.5 my-1",
                                line.startsWith("diff") && "text-blue-600 font-semibold",
                                line.startsWith("index") && "text-gray-500",
                                (line.startsWith("---") || line.startsWith("+++")) && "text-gray-600"
                              )}
                            >
                              {line}
                            </div>
                          ))}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}; 