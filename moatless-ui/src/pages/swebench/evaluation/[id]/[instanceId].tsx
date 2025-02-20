import { useParams, useNavigate } from "react-router-dom";
import { TrajectoryView } from "@/pages/trajectories/[id]";
import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { useEvaluationInstance } from "@/features/swebench/hooks/useEvaluationInstance";
import { InstanceList } from "../components/InstanceList";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/lib/components/ui/resizable";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/lib/components/ui/card";
import { Button } from "@/lib/components/ui/button";
import { ChevronLeft } from "lucide-react";

export function EvaluationInstancePage() {
  const { evaluationId, instanceId } = useParams<{ evaluationId: string; instanceId: string }>();
  const { data: evaluation, isError: evalError, error: evaluationError } = useEvaluation(evaluationId!);
  const { 
    data: trajectory,
    isError: trajectoryError,
    error: trajectoryErrorData,
    isLoading
  } = useEvaluationInstance(evaluationId!, instanceId!);

  if (!evaluationId || !instanceId) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Missing evaluation or instance ID</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (evalError) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {evaluationError instanceof Error ? evaluationError.message : "Failed to load evaluation data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!evaluation) {
    return (
      <div className="container mx-auto p-6">
        <Card>
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
    <div className="h-[calc(100vh-56px)]">
      <ResizablePanelGroup direction="horizontal" className="h-full border rounded-lg">
        <ResizableHandle className="bg-border hover:bg-ring" />
        <ResizablePanel defaultSize={85}>
          <TrajectoryView 
            trajectory={trajectory}
            isLoading={isLoading}
            isError={trajectoryError}
            error={trajectoryErrorData instanceof Error ? trajectoryErrorData : undefined}
          />
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
} 