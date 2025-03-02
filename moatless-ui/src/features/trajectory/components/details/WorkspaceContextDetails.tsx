import { Badge } from "@/lib/components/ui/badge.tsx";
import { Progress } from "@/lib/components/ui/progress.tsx";
import { cn } from "@/lib/utils.ts";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

interface WorkspaceContextDetailsProps {
  content: {
    files: Array<{
      file_path: string;
      tokens?: number;
      spans?: Array<{
        span_id: string;
        start_line: number;
        end_line: number;
        tokens?: number;
        pinned?: boolean;
      }>;
    }>;
    max_tokens?: number;
  };
}

export const WorkspaceContextDetails = ({
  content,
}: WorkspaceContextDetailsProps) => {
  // Calculate total tokens used
  const totalTokens = content.files.reduce(
    (sum, file) => sum + (file.tokens || 0),
    0,
  );
  const maxTokens = content.max_tokens || 0;
  const tokenPercentage = maxTokens
    ? Math.min(100, Math.round((totalTokens / maxTokens) * 100))
    : 0;

  // State to track expanded files and span groups
  const [expandedFiles, setExpandedFiles] = useState<Record<number, boolean>>(
    {},
  );

  // Toggle file expansion
  const toggleFileExpansion = (fileIdx: number) => {
    setExpandedFiles((prev) => ({
      ...prev,
      [fileIdx]: !prev[fileIdx],
    }));
  };

  // Group spans into categories for better organization
  const groupSpans = (
    spans: Array<{
      span_id: string;
      start_line: number;
      end_line: number;
      tokens?: number;
      pinned?: boolean;
    }>,
  ) => {
    // First show pinned spans, then sort by line number
    return [...spans].sort((a, b) => {
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      return a.start_line - b.start_line;
    });
  };

  // Determine if a file has many spans (more than 10)
  const hasLargeNumberOfSpans = (file: (typeof content.files)[0]) => {
    return file.spans && file.spans.length > 10;
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-gray-900">Files in Context</h3>

        {/* Token usage */}
        {maxTokens > 0 && (
          <div className="space-y-2 p-4 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-gray-700">Token Usage</h4>
              <span className="text-sm text-gray-600">
                {totalTokens} / {maxTokens} ({tokenPercentage}%)
              </span>
            </div>
            <Progress value={tokenPercentage} className="h-2" />
          </div>
        )}

        {/* Files list */}
        <div className="space-y-4">
          {content.files.map((file, idx) => (
            <div key={idx} className="rounded-lg border border-gray-200 p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  {hasLargeNumberOfSpans(file) && (
                    <button
                      onClick={() => toggleFileExpansion(idx)}
                      className="text-gray-500 hover:text-gray-700 focus:outline-none"
                    >
                      {expandedFiles[idx] ? (
                        <ChevronDown size={16} />
                      ) : (
                        <ChevronRight size={16} />
                      )}
                    </button>
                  )}
                  <span className="font-mono text-sm">{file.file_path}</span>
                </div>
                <div className="flex items-center gap-2">
                  {file.spans && (
                    <Badge variant="secondary" className="text-xs">
                      {file.spans.length} spans
                    </Badge>
                  )}
                  <Badge variant="outline">{file.tokens} tokens</Badge>
                </div>
              </div>

              {/* Spans */}
              {file.spans && file.spans.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xs font-medium text-gray-700">Spans</h4>
                    {hasLargeNumberOfSpans(file) && !expandedFiles[idx] && (
                      <span className="text-xs text-gray-500">
                        {file.spans.filter((s) => s.pinned).length} pinned,{" "}
                        {file.spans.length} total
                      </span>
                    )}
                  </div>

                  {/* Show only pinned spans when collapsed and file has many spans */}
                  {hasLargeNumberOfSpans(file) && !expandedFiles[idx] ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                      {groupSpans(file.spans)
                        .filter((span) => span.pinned)
                        .map((span) => (
                          <div
                            key={span.span_id}
                            className="flex items-center justify-between rounded border border-gray-200 px-3 py-2 text-xs bg-purple-50"
                          >
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {span.span_id}
                              </span>
                              <span className="rounded bg-purple-100 px-1 text-purple-600">
                                📌
                              </span>
                            </div>
                            <div className="text-gray-500">
                              <span>
                                Lines {span.start_line}-{span.end_line}
                              </span>
                              {span.tokens && (
                                <span className="ml-2">
                                  ({span.tokens} tokens)
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                    </div>
                  ) : (
                    <div
                      className={cn(
                        "grid gap-2",
                        hasLargeNumberOfSpans(file)
                          ? "grid-cols-1 md:grid-cols-3"
                          : "grid-cols-1 md:grid-cols-2",
                      )}
                    >
                      {groupSpans(file.spans).map((span) => (
                        <div
                          key={span.span_id}
                          className={cn(
                            "flex items-center justify-between rounded border border-gray-200 px-3 py-2 text-xs",
                            span.pinned && "bg-purple-50",
                          )}
                        >
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{span.span_id}</span>
                            {span.pinned && (
                              <span className="rounded bg-purple-100 px-1 text-purple-600">
                                📌
                              </span>
                            )}
                          </div>
                          <div className="text-gray-500">
                            <span>
                              Lines {span.start_line}-{span.end_line}
                            </span>
                            {span.tokens && (
                              <span className="ml-2">
                                ({span.tokens} tokens)
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Show expand/collapse button for files with many spans */}
                  {hasLargeNumberOfSpans(file) && (
                    <button
                      onClick={() => toggleFileExpansion(idx)}
                      className="mt-3 text-xs text-blue-600 hover:text-blue-800 focus:outline-none"
                    >
                      {expandedFiles[idx] ? "Collapse spans" : "Show all spans"}
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
