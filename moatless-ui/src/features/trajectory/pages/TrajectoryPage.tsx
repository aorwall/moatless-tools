import { TrajectoryViewer } from "@/features/trajectory/components/TrajectoryViewer.tsx";
import { useRetryTrajectory } from "@/features/trajectory/hooks/useRetryTrajectory";
import { useStartTrajectory } from "@/features/trajectory/hooks/useStartTrajectory";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { Card, CardContent } from "@/lib/components/ui/card";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { useParams } from "react-router-dom";
import { useRealtimeTrajectory, trajectoryKeys } from "@/features/trajectory/hooks/useRealtimeTrajectory";

export function TrajectoryPage() {
  const { projectId, trajectoryId } = useParams<{
    projectId: string;
    trajectoryId: string;
  }>();
  const queryClient = useQueryClient();

  if (!projectId || !trajectoryId) {
    throw new Error("No project ID or trajectory ID found in URL");
  }

  const {
    data: trajectory,
    isError: trajectoryError,
    error: trajectoryErrorData,
    isLoading,
  } = useRealtimeTrajectory(projectId, trajectoryId);

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <Card>
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
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {trajectoryErrorData instanceof Error
              ? trajectoryErrorData.message
              : "Failed to load trajectory data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const handleSuccess = () => {
    queryClient.invalidateQueries({
      queryKey: trajectoryKeys.detail(projectId, trajectoryId),
    });
  };

  const startTrajectory = useStartTrajectory({
    onSuccess: handleSuccess
  });

  const retryTrajectory = useRetryTrajectory({
    onSuccess: handleSuccess
  });

  const handleStart = async () => {
    await startTrajectory.mutateAsync({
      projectId,
      trajectoryId,
    });
  };

  const handleRetry = async () => {
    await retryTrajectory.mutateAsync({
      projectId,
      trajectoryId,
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
