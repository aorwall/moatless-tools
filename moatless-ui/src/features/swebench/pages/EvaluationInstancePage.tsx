import { useRealtimeEvaluationInstance } from "@/features/swebench/hooks/useRealtimeEvaluationInstance";
import { useRetryInstance } from "@/features/swebench/hooks/useRetryInstance";
import { useStartInstance } from "@/features/swebench/hooks/useStartInstance";
import { TrajectoryViewer } from "@/features/trajectory/components/TrajectoryViewer.tsx";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { Card, CardContent } from "@/lib/components/ui/card";
import { AlertCircle, Loader2 } from "lucide-react";
import { useParams } from "react-router-dom";

export function EvaluationInstancePage() {
  const { evaluationId, instanceId } = useParams<{
    evaluationId: string;
    instanceId: string;
  }>();

  if (!evaluationId || !instanceId) {
    throw new Error("No evaluation ID or instance ID found in URL");
  }

  const {
    data: trajectory,
    isError: trajectoryError,
    error: trajectoryErrorData,
    isLoading,
  } = useRealtimeEvaluationInstance(evaluationId, instanceId);

  const startInstance = useStartInstance();
  const retryInstance = useRetryInstance();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Card>
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading instance data...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (trajectoryError || !trajectory) {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {trajectoryErrorData instanceof Error
              ? trajectoryErrorData.message
              : "Failed to load data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <TrajectoryViewer
        trajectory={trajectory}
        handleStart={() =>
          startInstance.mutateAsync({
            evaluationName: evaluationId,
            instanceId: instanceId
          })
        }
        handleRetry={() =>
          retryInstance.mutateAsync({
            evaluationName: evaluationId,
            instanceId: instanceId
          })
        }
      />
    </div>
  );
}
