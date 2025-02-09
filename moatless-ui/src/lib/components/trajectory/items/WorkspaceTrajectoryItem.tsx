import { type FC } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";

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
}

export interface WorkspaceTimelineContent {
  updatedFiles?: Array<{
    file_path: string;
    is_new?: boolean;
    has_patch?: boolean;
    tokens?: number;
  }>;
  testResults?: Array<{
    name?: string;
    status: string;
    message?: string;
  }>;
  files?: File[];
  warnings?: string[];
}

export interface WorkspaceTrajectoryItemProps {
  content: WorkspaceTimelineContent;
  expandedState: boolean;
}

export const WorkspaceTrajectoryItem: FC<WorkspaceTrajectoryItemProps> = ({
  content,
  expandedState,
}) => {
  const isExpandable = !!(
    content.updatedFiles?.length ||
    content.testResults?.length ||
    content.files?.length
  );

  if (!expandedState) {
    return (
      <div className="text-xs text-gray-600">
        {content.updatedFiles?.length ? (
          <span>{content.updatedFiles.length} files updated</span>
        ) : content.testResults?.length ? (
          <span>{content.testResults.length} test results</span>
        ) : content.files?.length ? (
          <span>{content.files.length} files in context</span>
        ) : (
          <span>No workspace changes</span>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Updated Files Section */}
      {content.updatedFiles?.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-gray-700">
              Updated Files
            </div>
            <div className="text-xs text-gray-500">
              {content.updatedFiles.length} files
            </div>
          </div>
          <div className="space-y-2 rounded-md bg-gray-50 p-3">
            {content.updatedFiles.map((file, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-gray-700">
                      {file.file_path}
                    </span>
                    {file.is_new && (
                      <Badge variant="success" className="px-1.5 py-0.5">
                        new
                      </Badge>
                    )}
                    {file.has_patch && (
                      <Badge variant="default" className="px-1.5 py-0.5">
                        modified
                      </Badge>
                    )}
                    <span className="text-gray-500">{file.tokens} tokens</span>
                  </div>
                </div>

                {content.files?.find((f) => f.file_path === file.file_path)
                  ?.spans && (
                  <div className="flex flex-wrap gap-1.5 rounded border border-gray-200 p-2">
                    {content.files
                      .find((f) => f.file_path === file.file_path)
                      ?.spans?.map((span) => (
                        <div
                          key={span.span_id}
                          className="inline-flex items-center gap-1 rounded border border-gray-200 px-2 py-0.5 text-[10px] text-gray-700"
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

                {content.files?.find((f) => f.file_path === file.file_path)
                  ?.patch && (
                  <div className="max-w-full rounded bg-gray-100 p-2">
                    <pre className="overflow-x-auto whitespace-pre font-mono text-[10px]">
                      {content.files
                        .find((f) => f.file_path === file.file_path)
                        ?.patch?.split("\n")
                        .map((line, i) => (
                          <span
                            key={i}
                            className={cn(
                              line.startsWith("+") &&
                                "bg-green-50 text-green-700",
                              line.startsWith("-") && "bg-red-50 text-red-700",
                              line.startsWith("@") && "block text-purple-600",
                              !line.startsWith("+") &&
                                !line.startsWith("-") &&
                                !line.startsWith("@") &&
                                "text-gray-600",
                            )}
                          >
                            {line}
                            {"\n"}
                          </span>
                        ))}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Test Results Section */}
      {content.testResults?.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-gray-700">Test Results</div>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="text-gray-600">
              {content.testResults.length} total
            </Badge>
            <Badge
              variant="success"
              className={cn(
                !content.testResults.some((t) => t.status === "PASSED") &&
                  "bg-gray-50 text-gray-500",
              )}
            >
              {content.testResults.filter((t) => t.status === "PASSED").length}{" "}
              passed
            </Badge>
            <Badge
              variant="destructive"
              className={cn(
                !content.testResults.some((t) => t.status === "FAILED") &&
                  "bg-gray-50 text-gray-500",
              )}
            >
              {content.testResults.filter((t) => t.status === "FAILED").length}{" "}
              failed
            </Badge>
            <Badge
              variant="destructive"
              className={cn(
                !content.testResults.some((t) => t.status === "ERROR") &&
                  "bg-gray-50 text-gray-500",
              )}
            >
              {content.testResults.filter((t) => t.status === "ERROR").length}{" "}
              errors
            </Badge>
            <Badge
              variant="warning"
              className={cn(
                !content.testResults.some((t) => t.status === "SKIPPED") &&
                  "bg-gray-50 text-gray-500",
              )}
            >
              {content.testResults.filter((t) => t.status === "SKIPPED").length}{" "}
              skipped
            </Badge>
          </div>

          {content.testResults
            .filter((t) => t.status === "FAILED" || t.status === "ERROR")
            .map((test, idx) => (
              <div key={idx} className="space-y-2 rounded-md bg-gray-50 p-3">
                <div className="flex items-center justify-between text-xs">
                  <div className="font-medium text-gray-700">
                    {test.name || "Unnamed Test"}
                  </div>
                  <Badge
                    variant={
                      test.status === "FAILED" ? "destructive" : "destructive"
                    }
                    className="bg-red-100"
                  >
                    {test.status}
                  </Badge>
                </div>
                {test.message && (
                  <div className="w-full rounded bg-gray-100 p-2">
                    <pre className="overflow-x-auto whitespace-pre font-mono text-[10px]">
                      {test.message}
                    </pre>
                  </div>
                )}
              </div>
            ))}
        </div>
      )}
    </div>
  );
};
