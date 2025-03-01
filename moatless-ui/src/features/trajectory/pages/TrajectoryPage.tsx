import { useParams } from "react-router-dom";
import { TrajectoryViewer } from "@/lib/components/trajectory/TrajectoryViewer";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/lib/components/ui/card";
import { useGetTrajectory } from "@/features/trajectory/hooks/useGetTrajectory";



export function TrajectoryPage() {
  const { projectId, trajectoryId } = useParams();

  if (!projectId || !trajectoryId) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>No project id or trajectory id found</AlertDescription>
        </Alert>
      </div>
    );
  }

  const { data: trajectory, isLoading, isError, error } = useGetTrajectory(projectId, trajectoryId);

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

  if (isError || !trajectory) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load trajectory data"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="h-full w-full">
      <TrajectoryViewer trajectory={trajectory} />
    </div>
  );
}
