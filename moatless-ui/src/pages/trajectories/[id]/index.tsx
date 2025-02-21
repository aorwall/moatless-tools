import { useParams } from "react-router-dom";
import { useGetTrajectory } from "@/lib/hooks/useGetTrajectory";
import { TrajectoryViewer } from "@/lib/components/trajectory/TrajectoryViewer";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle } from "lucide-react";

export function TrajectoryPage() {
  const { trajectoryId } = useParams();

  if (!trajectoryId) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>No trajectory id found</AlertDescription>
        </Alert>
      </div>
    );
  }

  const { data: trajectory, isLoading, isError, error } = useGetTrajectory(trajectoryId);

  return (
    <div className="container mx-auto p-6">
      <TrajectoryViewer 
        trajectory={trajectory} 
        isLoading={isLoading} 
        isError={isError} 
        error={error as Error | undefined} 
      />
    </div>
  );
}
