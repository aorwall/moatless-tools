import { type FC } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { FileCode, FilePlus, FileEdit } from "lucide-react";
import { countPatchChanges, getFileName } from "@/lib/hooks/useFileUtils";

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

export interface WorkspaceFilesDetailsProps {
  content: WorkspaceFilesTimelineContent;
}

export const WorkspaceFilesDetails: FC<WorkspaceFilesDetailsProps> = ({
  content,
}) => {
  // Count new vs modified files
  const newFiles = content.updatedFiles?.filter(f => f.is_new)?.length || 0;
  const modifiedFiles = content.updatedFiles?.filter(f => !f.is_new && f.patch)?.length || 0;
  
  return (
    <div className="space-y-4">
      {/* Summary Section */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Updated Files</h3>
        <div className="flex items-center gap-2">
          {newFiles > 0 && (
            <div className="flex items-center gap-1 text-xs">
              <FilePlus size={14} className="text-green-600" />
              <span className="text-green-700">{newFiles} new</span>
            </div>
          )}
          {modifiedFiles > 0 && (
            <div className="flex items-center gap-1 text-xs">
              <FileEdit size={14} className="text-blue-600" />
              <span className="text-blue-700">{modifiedFiles} modified</span>
            </div>
          )}
        </div>
      </div>

      {/* Files Tabs */}
      {content.updatedFiles && content.updatedFiles.length > 0 && (
        <Tabs defaultValue={content.updatedFiles[0].file_path} className="w-full">
          <TabsList className="w-full overflow-x-auto flex-nowrap whitespace-nowrap">
            {content.updatedFiles.map((file, idx) => (
              <TabsTrigger 
                key={idx} 
                value={file.file_path}
                className="flex items-center gap-1.5"
              >
                {file.is_new ? (
                  <FilePlus size={12} className="text-green-600" />
                ) : (
                  <FileCode size={12} className="text-blue-600" />
                )}
                <span className="truncate max-w-[150px]">
                  {getFileName(file.file_path)}
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
          
          {content.updatedFiles.map((file, idx) => {
            const fullFile = file;
            const { additions, deletions } = countPatchChanges(fullFile?.patch || '');
            
            return (
              <TabsContent key={idx} value={file.file_path} className="space-y-4 mt-4">
                <div className="flex items-center justify-between">
                  <div className="font-mono text-sm text-gray-700 truncate max-w-[70%]">
                    {file.file_path}
                  </div>
                  <div className="flex items-center gap-2">
                    {file.is_new ? (
                      <Badge variant="default" className="bg-green-100 text-green-800">
                        New File
                      </Badge>
                    ) : (
                      <Badge variant="default" className="bg-blue-100 text-blue-800">
                        Modified
                      </Badge>
                    )}
                    
                    {fullFile?.patch && (
                      <div className="flex items-center gap-1 text-xs">
                        <span className="text-green-700">+{additions}</span>
                        <span>/</span>
                        <span className="text-red-700">-{deletions}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                {/* File Patch */}
                {fullFile?.patch && (
                  <div className="border rounded-md overflow-hidden">
                    <div className="bg-gray-100 px-3 py-1.5 border-b flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-700">Patch</span>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">
                          +{additions}
                        </Badge>
                        <Badge variant="outline" className="text-xs bg-red-50 text-red-700 border-red-200">
                          -{deletions}
                        </Badge>
                      </div>
                    </div>
                    <div className="max-h-[400px] overflow-auto bg-gray-50">
                      <pre className="p-3 text-xs font-mono whitespace-pre">
                        {fullFile.patch.split("\n").map((line, i) => (
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
                  </div>
                )}
                
                {/* Spans */}
                {fullFile?.spans && fullFile.spans.length > 0 && (
                  <div className="border rounded-md overflow-hidden mt-4">
                    <div className="bg-gray-100 px-3 py-1.5 border-b">
                      <span className="text-xs font-medium text-gray-700">Spans</span>
                    </div>
                    <div className="p-3 bg-gray-50">
                      <div className="flex flex-wrap gap-2">
                        {fullFile.spans.map((span) => (
                          <div
                            key={span.span_id}
                            className={cn(
                              "inline-flex items-center gap-1.5 rounded border border-gray-200 px-2 py-1 text-xs text-gray-700",
                              span.pinned && "bg-purple-50 border-purple-200"
                            )}
                          >
                            <span className="font-medium">{span.span_id}</span>
                            <span className="text-gray-500">
                              ({span.start_line}-{span.end_line})
                            </span>
                            {span.pinned && (
                              <span className="rounded bg-purple-100 px-1 text-purple-600">
                                📌
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </TabsContent>
            );
          })}
        </Tabs>
      )}
    </div>
  );
}; 