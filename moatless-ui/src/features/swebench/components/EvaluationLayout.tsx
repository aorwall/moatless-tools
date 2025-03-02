import { Outlet, useParams, useNavigate } from "react-router-dom";
import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { InstanceList } from "./InstanceList";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle, Loader2, ChevronLeft } from "lucide-react";
import { Card, CardContent } from "@/lib/components/ui/card";
import { Button } from "@/lib/components/ui/button";

export function EvaluationLayout() {
  const { evaluationId, instanceId } = useParams();
  const navigate = useNavigate();
  const { data: evaluation, isError, error } = useEvaluation(evaluationId!);

  if (!evaluationId) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Missing evaluation ID</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error instanceof Error
              ? error.message
              : "Failed to load evaluation data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Card className="max-w-md">
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading evaluation data...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-56px)] w-full">
      <ResizablePanelGroup direction="horizontal" className="h-full">
        <ResizablePanel defaultSize={15} minSize={0}>
          {instanceId && (
            <div className="p-4 border-b">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate(`/swebench/evaluation/${evaluationId}`)}
                className="flex items-center gap-2"
              >
                <ChevronLeft className="h-4 w-4" />
                Back to Evaluation
              </Button>
            </div>
          )}
          <InstanceList
            evaluation={evaluation}
            selectedInstanceId={instanceId}
          />
        </ResizablePanel>
        <ResizableHandle className="bg-border hover:bg-ring" />
        <ResizablePanel defaultSize={85}>
          <Outlet />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
