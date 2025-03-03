import { useEvaluation } from "@/features/swebench/hooks/useEvaluation";
import { useEvaluationInstance } from "@/features/swebench/hooks/useEvaluationInstance";
import { useRetryInstance } from "@/features/swebench/hooks/useRetryInstance";
import { useStartInstance } from "@/features/swebench/hooks/useStartInstance";
import { TrajectoryViewer } from "@/features/trajectory/components/TrajectoryViewer.tsx";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { Card, CardContent } from "@/lib/components/ui/card";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useParams } from "react-router-dom";
import {useTrajectorySubscription} from "@/features/trajectory/hooks/useTrajectorySubscription.ts";

export function EvaluationInstancePage() {
  const { evaluationId, instanceId } = useParams<{
    evaluationId: string;
    instanceId: string;
  }>();
  const queryClient = useQueryClient();

  if (!evaluationId || !instanceId) {
    return (
        <div className="container mx-auto p-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              No evaluation ID or instance ID found in URL
            </AlertDescription>
          </Alert>
        </div>
    );
  }

  const {
    data: trajectory,
    isError: trajectoryError,
    error: trajectoryErrorData,
    isLoading,
  } = useEvaluationInstance(evaluationId!, instanceId!);

  if (isLoading) {
    return (
        <div className="container mx-auto p-6">
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

  if (evalError || trajectoryError || !evaluation || !trajectory) {
    return (
        <div className="container mx-auto p-6">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {evaluationError instanceof Error
                  ? evaluationError.message
                  : trajectoryErrorData instanceof Error
                      ? trajectoryErrorData.message
                      : "Failed to load data"}
            </AlertDescription>
          </Alert>
        </div>
    );
  }

  const handleSuccess = () => {
    queryClient.invalidateQueries({
      queryKey: ["evaluation", evaluationId, "instance", instanceId],
    });
  };

  useTrajectorySubscription(trajectory.id, trajectory.project_id, {
    onEvent: (message) => {
      if (message.type === "event") {
        queryClient.invalidateQueries({
          queryKey: ["evaluation", evaluationId, "instance", instanceId],
        });
      }
    },
    showToasts: process.env.NODE_ENV === "development"
  });

  const startInstance = useStartInstance({
    onSuccess: handleSuccess
  });

  const retryInstance = useRetryInstance({
    onSuccess: handleSuccess
  });

  const handleStart = async () => {
    await startInstance.mutateAsync({
      evaluationName: evaluationId,
      instanceId: instanceId
    });
  };

  const handleRetry = async () => {
    await retryInstance.mutateAsync({
      evaluationName: evaluationId,
      instanceId: instanceId
    });
  };

  return (
    <div className="h-full w-full">
      <TrajectoryViewer
        trajectory={trajectory}
        handleStart={handleStart}
        handleRetry={handleRetry}
      />
    </div>
  );
}
