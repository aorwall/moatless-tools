import { type FC } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/lib/components/ui/badge";
import { CheckCircle, AlertCircle, AlertTriangle, Clock } from "lucide-react";

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
  expandedState: boolean;
}

export const WorkspaceTestsTrajectoryItem: FC<WorkspaceTestsTrajectoryItemProps> = ({
  content,
  expandedState,
}) => {
  // Calculate total test results across all test files
  const allTestResults = content.test_files?.flatMap(f => f.test_results) ?? [];
  const testStatusCounts = {
    total: allTestResults.length,
    passed: allTestResults.filter(t => t.status === "PASSED").length,
    failed: allTestResults.filter(t => t.status === "FAILED").length,
    error: allTestResults.filter(t => t.status === "ERROR").length,
    skipped: allTestResults.filter(t => t.status === "SKIPPED").length,
  };

  if (!expandedState) {
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
                <span className="text-red-700">{testStatusCounts.failed + testStatusCounts.error}</span>
              </div>
            )}
            
            {testStatusCounts.skipped > 0 && (
              <div className="flex items-center gap-1">
                <Clock size={12} className="text-gray-500" />
                <span className="text-gray-600">{testStatusCounts.skipped}</span>
              </div>
            )}
            
            <span className="text-gray-500">
              of {testStatusCounts.total} tests in {content.test_files.length} files
            </span>
          </div>
        ) : (
          <span className="text-gray-600">No test results</span>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Test Results Section */}
      {content.test_files && content.test_files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-xs font-medium text-gray-700">Test Results</div>
            <div className="text-xs text-gray-500">
              {content.test_files.length} files
            </div>
          </div>
          
          {/* Test summary badges with icons */}
          <div className="flex flex-wrap gap-2 p-2 bg-gray-50 rounded-md">
            <div className="flex items-center gap-1.5">
              <CheckCircle size={14} className={cn(
                "text-green-600",
                testStatusCounts.passed === 0 && "text-gray-400"
              )} />
              <Badge
                variant="default"
                className={cn(
                  "bg-green-100 text-green-800",
                  testStatusCounts.passed === 0 && "bg-gray-50 text-gray-500"
                )}
              >
                {testStatusCounts.passed} passed
              </Badge>
            </div>
            
            <div className="flex items-center gap-1.5">
              <AlertCircle size={14} className={cn(
                "text-red-600",
                testStatusCounts.failed === 0 && "text-gray-400"
              )} />
              <Badge
                variant="destructive"
                className={cn(
                  testStatusCounts.failed === 0 && "bg-gray-50 text-gray-500"
                )}
              >
                {testStatusCounts.failed} failed
              </Badge>
            </div>
            
            <div className="flex items-center gap-1.5">
              <AlertTriangle size={14} className={cn(
                "text-amber-600",
                testStatusCounts.error === 0 && "text-gray-400"
              )} />
              <Badge
                variant="destructive"
                className={cn(
                  "bg-amber-100 text-amber-800",
                  testStatusCounts.error === 0 && "bg-gray-50 text-gray-500"
                )}
              >
                {testStatusCounts.error} errors
              </Badge>
            </div>
            
            <div className="flex items-center gap-1.5">
              <Clock size={14} className={cn(
                "text-gray-600",
                testStatusCounts.skipped === 0 && "text-gray-400"
              )} />
              <Badge
                variant="secondary"
                className={cn(
                  testStatusCounts.skipped === 0 && "bg-gray-50 text-gray-500"
                )}
              >
                {testStatusCounts.skipped} skipped
              </Badge>
            </div>
          </div>

          {content.test_files.map((testFile, fileIdx) => {
            // Count test results by status for this file
            const fileTestCounts = {
              total: testFile.test_results.length,
              passed: testFile.test_results.filter(t => t.status === "PASSED").length,
              failed: testFile.test_results.filter(t => t.status === "FAILED").length,
              error: testFile.test_results.filter(t => t.status === "ERROR").length,
              skipped: testFile.test_results.filter(t => t.status === "SKIPPED").length,
            };
            
            return (
              <div key={fileIdx} className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <div className="font-mono text-gray-700">{testFile.file_path}</div>
                  <div className="flex items-center gap-2">
                    {fileTestCounts.passed > 0 && (
                      <span className="flex items-center gap-1 text-green-700">
                        <CheckCircle size={12} />
                        {fileTestCounts.passed}
                      </span>
                    )}
                    {fileTestCounts.failed > 0 && (
                      <span className="flex items-center gap-1 text-red-700">
                        <AlertCircle size={12} />
                        {fileTestCounts.failed}
                      </span>
                    )}
                    {fileTestCounts.error > 0 && (
                      <span className="flex items-center gap-1 text-amber-700">
                        <AlertTriangle size={12} />
                        {fileTestCounts.error}
                      </span>
                    )}
                  </div>
                </div>
                
                {testFile.test_results
                  .filter((t) => t.status === "FAILED" || t.status === "ERROR")
                  .map((test, testIdx) => (
                    <div key={testIdx} className="space-y-2 rounded-md bg-gray-50 p-3">
                      <div className="flex items-center justify-between text-xs">
                        <Badge
                          variant={test.status === "FAILED" ? "destructive" : "outline"}
                          className={cn(
                            test.status === "ERROR" && "bg-amber-100 text-amber-800 border-amber-200"
                          )}
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
            );
          })}
        </div>
      )}
    </div>
  );
}; 