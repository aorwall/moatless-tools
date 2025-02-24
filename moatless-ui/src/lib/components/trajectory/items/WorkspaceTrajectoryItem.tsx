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
  show_all_spans?: boolean;
}

export interface WorkspaceTimelineContent {
  max_tokens?: number;
  updatedFiles?: Array<{
    file_path: string;
    is_new?: boolean;
    has_patch?: boolean;
    tokens?: number;
  }>;
  test_files?: Array<{
    file_path: string;
    test_results: Array<{
      status: string;
      message?: string;
    }>;
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
    content.test_files?.length ||
    content.files?.length
  );

  if (!expandedState) {
    return (
      <div className="text-xs text-gray-600">
        {content.updatedFiles?.length ? (
          <span>{content.updatedFiles.length} files updated</span>
        ) : content.test_files?.length ? (
          <span>{content.test_files.length} test files</span>
        ) : content.files?.length ? (
          <span>{content.files.length} files in context</span>
        ) : (
          <span>No workspace changes</span>
        )}
      </div>
    );
  }

  // Calculate total test results across all test files
  const allTestResults = content.test_files?.flatMap(f => f.test_results) ?? [];
  const testStatusCounts = {
    total: allTestResults.length,
    passed: allTestResults.filter(t => t.status === "PASSED").length,
    failed: allTestResults.filter(t => t.status === "FAILED").length,
    error: allTestResults.filter(t => t.status === "ERROR").length,
    skipped: allTestResults.filter(t => t.status === "SKIPPED").length,
  };

  return (
    <div className="space-y-4">
      {/* Updated Files Section */}
      {content.updatedFiles && content.updatedFiles.length > 0 && (
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
                      <Badge variant="default" className="px-1.5 py-0.5 bg-green-100 text-green-800">
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
      {content.test_files && content.test_files.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-gray-700">Test Results</div>
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="text-gray-600">
              {testStatusCounts.total} total
            </Badge>
            <Badge
              variant="default"
              className={cn(
                "bg-green-100 text-green-800",
                testStatusCounts.passed === 0 && "bg-gray-50 text-gray-500"
              )}
            >
              {testStatusCounts.passed} passed
            </Badge>
            <Badge
              variant="destructive"
              className={cn(
                testStatusCounts.failed === 0 && "bg-gray-50 text-gray-500"
              )}
            >
              {testStatusCounts.failed} failed
            </Badge>
            <Badge
              variant="destructive"
              className={cn(
                testStatusCounts.error === 0 && "bg-gray-50 text-gray-500"
              )}
            >
              {testStatusCounts.error} errors
            </Badge>
            <Badge
              variant="secondary"
              className={cn(
                testStatusCounts.skipped === 0 && "bg-gray-50 text-gray-500"
              )}
            >
              {testStatusCounts.skipped} skipped
            </Badge>
          </div>

          {content.test_files.map((testFile, fileIdx) => (
            <div key={fileIdx} className="space-y-2">
              <div className="text-xs font-medium text-gray-700">{testFile.file_path}</div>
              {testFile.test_results
                .filter((t) => t.status === "FAILED" || t.status === "ERROR")
                .map((test, testIdx) => (
                  <div key={testIdx} className="space-y-2 rounded-md bg-gray-50 p-3">
                    <div className="flex items-center justify-between text-xs">
                      <Badge
                        variant="destructive"
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
          ))}
        </div>
      )}
    </div>
  );
};
