import { Button } from "@/lib/components/ui/button.tsx";
import { JsonViewer } from "@/lib/components/ui/json-viewer.tsx";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/lib/components/ui/tooltip.tsx";
import { useExecuteNode } from "@/lib/hooks/useExecuteNode.ts";
import { Trajectory } from "@/lib/types/trajectory.ts";
import { Info, Play } from "lucide-react";
import { useState } from "react";

interface ExecutionResult {
  data?: any;
  error?: string;
  timestamp: number;
}

interface ActionDetailsProps {
  content: any;
  nodeId: number;
  trajectory: Trajectory;
}

export const ActionDetails = ({
  content,
  nodeId,
  trajectory,
}: ActionDetailsProps) => {
  const [executionResults, setExecutionResults] = useState<ExecutionResult[]>(
    [],
  );
  const executeNode = useExecuteNode();

  const handleExecute = () => {
    executeNode.mutate({
      nodeId,
      trajectory,
      onSuccess: (data) => {
        setExecutionResults((prev) => [
          ...prev,
          {
            data,
            timestamp: Date.now(),
          },
        ]);
      },
      onError: (error) => {
        setExecutionResults((prev) => [
          ...prev,
          {
            error:
              error instanceof Error
                ? error.message
                : "An unexpected error occurred",
            timestamp: Date.now(),
          },
        ]);
      },
    });
  };

  return (
    <TooltipProvider>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-sm text-gray-600">
              Action Details
            </h3>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-gray-400 cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  You can execute this action again to troubleshoot or verify
                  its behavior.
                </p>
              </TooltipContent>
            </Tooltip>
          </div>
          <Button
            onClick={handleExecute}
            disabled={executeNode.isPending}
            variant="ghost"
            size="sm"
            className="text-gray-600 hover:text-gray-900"
          >
            <Play className="h-4 w-4 mr-2" />
            {executeNode.isPending ? "Executing..." : "Execute Again"}
          </Button>
        </div>

        <div className="overflow-x-auto rounded-md bg-gray-50 p-4">
          <div className="min-w-[300px]">
            <div className="font-mono text-sm">
              <JsonViewer data={content} />
            </div>
          </div>
        </div>

        {executionResults.length > 0 && (
          <div className="space-y-3 border-t pt-4 mt-6">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold text-sm text-gray-600">
                Execution Results
              </h3>
              <span className="text-xs text-gray-500">(newest first)</span>
            </div>
            {executionResults
              .map((result, index) => (
                <div
                  key={result.timestamp}
                  className="overflow-x-auto rounded-md bg-gray-50 p-4"
                >
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="text-xs font-medium text-gray-500">
                      {new Date(result.timestamp).toLocaleString()}
                    </h4>
                  </div>
                  {result.error ? (
                    <div className="text-red-600 font-mono text-sm bg-red-50 p-3 rounded border border-red-100">
                      {result.error}
                    </div>
                  ) : (
                    <div className="font-mono text-sm">
                      <JsonViewer data={result.data} />
                    </div>
                  )}
                </div>
              ))
              .reverse()}
          </div>
        )}
      </div>
    </TooltipProvider>
  );
};
