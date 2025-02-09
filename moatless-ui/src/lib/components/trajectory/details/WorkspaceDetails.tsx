import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";

interface WorkspaceDetailsProps {
  content: {
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
    files?: Array<{
      file_path: string;
      is_new?: boolean;
      was_edited?: boolean;
      tokens?: number;
      patch?: string;
      spans?: Array<{
        span_id: string;
        start_line: number;
        end_line: number;
        tokens?: number;
        pinned?: boolean;
      }>;
    }>;
    warnings?: string[];
  };
}

export const WorkspaceDetails = ({ content }: WorkspaceDetailsProps) => {
  return (
    <div className="space-y-6">
      {/* Files Section */}
      {content.updatedFiles?.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-900">Updated Files</h3>
          <div className="space-y-4">
            {content.updatedFiles.map((file, idx) => (
              <div key={idx} className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-mono text-sm">{file.file_path}</span>
                  <div className="flex gap-2">
                    {file.is_new && <Badge>New</Badge>}
                    {file.has_patch && (
                      <Badge variant="secondary">Modified</Badge>
                    )}
                  </div>
                </div>

                {content.files?.find((f) => f.file_path === file.file_path)
                  ?.patch && (
                  <pre className="mt-4 overflow-x-auto rounded-md bg-gray-50 p-4 text-sm">
                    {
                      content.files.find((f) => f.file_path === file.file_path)
                        ?.patch
                    }
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Test Results Section */}
      {content.testResults?.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-900">Test Results</h3>
          <div className="space-y-4">
            {content.testResults.map((test, idx) => (
              <div key={idx} className="rounded-lg border border-gray-200 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">
                    {test.name || "Unnamed Test"}
                  </span>
                  <Badge
                    variant={
                      test.status === "PASSED"
                        ? "default"
                        : test.status === "FAILED"
                          ? "destructive"
                          : "secondary"
                    }
                  >
                    {test.status}
                  </Badge>
                </div>
                {test.message && (
                  <pre className="mt-2 whitespace-pre-wrap rounded-md bg-gray-50 p-3 text-sm">
                    {test.message}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
