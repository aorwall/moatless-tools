import { AlertCircle, CheckCircle, Clock } from "lucide-react";
import { type FC } from "react";

interface TestResult {
  status: string;
  message?: string;
}

interface TestFile {
  file_path: string;
  test_results: TestResult[];
}

export interface WorkspaceTestsTimelineContent {
  test_files: TestFile[];
}

export interface WorkspaceTestsTrajectoryItemProps {
  content: WorkspaceTestsTimelineContent;
}

export const WorkspaceTestsTrajectoryItem: FC<
  WorkspaceTestsTrajectoryItemProps
> = ({ content }) => {
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

  return (
    <div className="text-xs">
      {testStatusCounts.total > 0 ? (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <CheckCircle size={12} className="text-green-600" />
            <span className="text-green-700">{testStatusCounts.passed}</span>
          </div>

          {(testStatusCounts.failed > 0 || testStatusCounts.error > 0) && (
            <div className="flex items-center gap-1">
              <AlertCircle size={12} className="text-red-600" />
              <span className="text-red-700">
                {testStatusCounts.failed + testStatusCounts.error}
              </span>
            </div>
          )}

          {testStatusCounts.skipped > 0 && (
            <div className="flex items-center gap-1">
              <Clock size={12} className="text-gray-500" />
              <span className="text-gray-600">{testStatusCounts.skipped}</span>
            </div>
          )}

          <span className="text-gray-500">
            of {testStatusCounts.total} tests in {content.test_files.length}{" "}
            files
          </span>
        </div>
      ) : (
        <span className="text-gray-600">No test results</span>
      )}
    </div>
  );
};
