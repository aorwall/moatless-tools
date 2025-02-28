import { useParams } from "react-router-dom";
import { TrajectoryViewer } from "@/lib/components/trajectory/TrajectoryViewer";
import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { useEvaluationInstance } from "@/features/swebench/hooks/useEvaluationInstance";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/lib/components/ui/card";
import { useStartInstance } from "@/features/swebench/hooks/useStartInstance";

export function EvaluationInstancePage() {
  const { evaluationId, instanceId } = useParams<{ evaluationId: string; instanceId: string }>();

  const { data: evaluation, isError: evalError, error: evaluationError } = useEvaluation(evaluationId!);
  const { 
    data: trajectory,
    isError: trajectoryError,
    error: trajectoryErrorData,
    isLoading
  } = useEvaluationInstance(evaluationId!, instanceId!);
  
  const startInstanceMutation = useStartInstance();

  const handleStartInstance = () => {
    if (evaluationId && instanceId) {
      startInstanceMutation.mutate({
        evaluationName: evaluationId,
        instanceId: instanceId
      });
    }
  };

  if (!evaluationId || !instanceId) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Missing evaluation or instance ID</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (evalError) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
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

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Card className="max-w-md">
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading trajectory data...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (trajectoryError || !trajectory) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {trajectoryErrorData instanceof Error ? trajectoryErrorData.message : "Failed to load trajectory data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <TrajectoryViewer 
        trajectory={trajectory} 
        startInstance={handleStartInstance}
      />
    </div>
  );
} 