import { Badge } from "@/lib/components/ui/badge";
import { cn } from "@/lib/utils";

interface WorkspaceDetailsProps {
  content: {
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
      show_all_spans?: boolean;
    }>;
    warnings?: string[];
  };
}

export const WorkspaceDetails = ({ content }: WorkspaceDetailsProps) => {
  console.log(content);
  return (
    <div className="space-y-6">
      {/* Files Section */}
      {content.updatedFiles && content.updatedFiles.length > 0 && (
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
      {content.test_files && content.test_files.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-gray-900">Test Results</h3>
          <div className="space-y-4">
            {content.test_files.map((testFile, idx) => (
              <div key={idx} className="rounded-lg border border-gray-200 p-4">
                <div className="font-medium mb-3">{testFile.file_path}</div>
                {testFile.test_results.map((test, testIdx) => (
                  <div key={testIdx} className="mt-2">
                    <div className="flex items-center gap-2">
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
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
