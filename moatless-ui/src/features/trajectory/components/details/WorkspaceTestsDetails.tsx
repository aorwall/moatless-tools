import { Badge } from "@/lib/components/ui/badge.tsx";
import { cn } from "@/lib/utils.ts";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

interface WorkspaceTestsDetailsProps {
  content: {
    test_files: Array<{
      file_path: string;
      test_results: Array<{
        status: string;
        message?: string;
      }>;
    }>;
  };
}

export const WorkspaceTestsDetails = ({
  content,
}: WorkspaceTestsDetailsProps) => {
  // State to track which file sections are expanded
  const [expandedFiles, setExpandedFiles] = useState<Record<number, boolean>>(
    {},
  );

  // Toggle expansion state for a file
  const toggleFileExpansion = (fileIdx: number) => {
    setExpandedFiles((prev) => ({
      ...prev,
      [fileIdx]: !prev[fileIdx],
    }));
  };

  // Calculate total test results across all test files
  const allTestResults =
    content.test_files?.flatMap((f) => f.test_results) ?? [];
  const testStatusCounts = {
    total: allTestResults.length,
    passed: allTestResults.filter((t) => t.status === "PASSED").length,
    failed: allTestResults.filter((t) => t.status === "FAILED").length,
    error: allTestResults.filter((t) => t.status === "ERROR").length,
    skipped: allTestResults.filter((t) => t.status === "SKIPPED").length,
  };

  // Helper function to calculate test counts for a single file
  const calculateFileTestCounts = (
    testResults: Array<{ status: string; message?: string }>,
  ) => {
    return {
      total: testResults.length,
      passed: testResults.filter((t) => t.status === "PASSED").length,
      failed: testResults.filter((t) => t.status === "FAILED").length,
      error: testResults.filter((t) => t.status === "ERROR").length,
      skipped: testResults.filter((t) => t.status === "SKIPPED").length,
    };
  };

  // Helper function to sort test results - failed and errors first, then skipped, then passed
  const sortTestResults = (
    tests: Array<{ status: string; message?: string }>,
  ) => {
    return [...tests].sort((a, b) => {
      // Define priority order: FAILED/ERROR > SKIPPED > PASSED
      const getPriority = (status: string) => {
        if (status === "FAILED" || status === "ERROR") return 0;
        if (status === "SKIPPED") return 1;
        return 2; // PASSED
      };

      return getPriority(a.status) - getPriority(b.status);
    });
  };

  // Helper function to get test status icon color
  const getStatusColor = (counts: typeof testStatusCounts) => {
    if (counts.failed > 0 || counts.error > 0) return "text-red-500";
    if (counts.skipped > 0 && counts.passed === 0) return "text-yellow-500";
    if (counts.passed > 0) return "text-green-500";
    return "text-gray-500";
  };

  // Reusable test summary component
  const TestSummaryBadges = ({
    counts,
    compact = false,
  }: {
    counts: typeof testStatusCounts;
    compact?: boolean;
  }) => (
    <div className={cn("flex flex-wrap gap-2", compact && "gap-1.5")}>
      <Badge
        variant="outline"
        className={cn("text-gray-600", compact && "text-xs px-1.5 py-0")}
      >
        {counts.total} total
      </Badge>
      <Badge
        variant="default"
        className={cn(
          "bg-green-100 text-green-800",
          counts.passed === 0 && "bg-gray-50 text-gray-500",
          compact && "text-xs px-1.5 py-0",
        )}
      >
        {counts.passed} passed
      </Badge>
      <Badge
        variant="destructive"
        className={cn(
          counts.failed === 0 && "bg-gray-50 text-gray-500",
          compact && "text-xs px-1.5 py-0",
        )}
      >
        {counts.failed} failed
      </Badge>
      <Badge
        variant="destructive"
        className={cn(
          counts.error === 0 && "bg-gray-50 text-gray-500",
          compact && "text-xs px-1.5 py-0",
        )}
      >
        {counts.error} errors
      </Badge>
      <Badge
        variant="secondary"
        className={cn(
          counts.skipped === 0 && "bg-gray-50 text-gray-500",
          compact && "text-xs px-1.5 py-0",
        )}
      >
        {counts.skipped} skipped
      </Badge>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-gray-900">Test Results</h3>

        {/* Overall test summary */}
        <div className="p-4 rounded-lg border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Summary</h4>
          <TestSummaryBadges counts={testStatusCounts} />
        </div>

        {/* Files summary section */}
        <div className="p-4 rounded-lg border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Test Files ({content.test_files.length})
          </h4>
          <div className="space-y-2">
            {content.test_files.map((testFile, fileIdx) => {
              const fileTestCounts = calculateFileTestCounts(
                testFile.test_results,
              );
              const isExpanded = expandedFiles[fileIdx] || false;
              const statusColor = getStatusColor(fileTestCounts);

              return (
                <div
                  key={fileIdx}
                  className="rounded-lg border border-gray-200 overflow-hidden"
                >
                  {/* File header - always visible */}
                  <div
                    className="px-4 py-3 flex flex-col cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => toggleFileExpansion(fileIdx)}
                  >
                    <div className="flex items-center mb-2">
                      <div className="mr-2 flex-shrink-0">
                        {isExpanded ? (
                          <ChevronDown className="h-4 w-4 text-gray-500" />
                        ) : (
                          <ChevronRight className="h-4 w-4 text-gray-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center">
                          <span
                            className={cn(
                              "w-2.5 h-2.5 rounded-full flex-shrink-0 mr-2",
                              statusColor,
                            )}
                          ></span>
                          <span className="font-medium text-sm truncate pr-2">
                            {testFile.file_path}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="ml-6">
                      <TestSummaryBadges
                        counts={fileTestCounts}
                        compact={true}
                      />
                    </div>
                  </div>

                  {/* Expandable test details */}
                  {isExpanded && (
                    <div className="border-t border-gray-200 p-4 space-y-3 bg-gray-50">
                      {sortTestResults(testFile.test_results).map(
                        (test, testIdx) => (
                          <div
                            key={testIdx}
                            className={cn(
                              "rounded-md p-3",
                              test.status === "PASSED" && "bg-green-50",
                              (test.status === "FAILED" ||
                                test.status === "ERROR") &&
                              "bg-red-50",
                              test.status === "SKIPPED" && "bg-gray-50",
                            )}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <Badge
                                variant={
                                  test.status === "PASSED"
                                    ? "default"
                                    : test.status === "FAILED" ||
                                      test.status === "ERROR"
                                      ? "destructive"
                                      : "secondary"
                                }
                                className={cn(
                                  test.status === "PASSED" &&
                                  "bg-green-100 text-green-800",
                                  test.status === "SKIPPED" &&
                                  "bg-gray-200 text-gray-800",
                                )}
                              >
                                {test.status}
                              </Badge>
                            </div>
                            {test.message && (
                              <pre className="overflow-x-auto whitespace-pre-wrap text-xs font-mono bg-white rounded p-2 border border-gray-200">
                                {test.message}
                              </pre>
                            )}
                          </div>
                        ),
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
