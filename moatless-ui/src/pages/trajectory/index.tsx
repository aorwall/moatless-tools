import { useSearchParams } from "react-router-dom";
import { TrajectoryHeader } from "@/pages/trajectory/components/TrajectoryHeader";
import { TrajectoryUpload } from "@/pages/trajectory/components/TrajectoryUpload";
import { TrajectoryViewer } from "@/lib/components/trajectory/TrajectoryViewer";
import { useGetTrajectory } from "@/lib/hooks/useGetTrajectory";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/lib/components/ui/card";

export function Trajectory() {
  const [searchParams, setSearchParams] = useSearchParams();
  const path = searchParams.get("path");
  const { data: trajectory, isLoading, isError, error } = useGetTrajectory(path || "");

  const handleLoadTrajectory = (path: string) => {
    setSearchParams({ path });
  };

  return (
    <div className="flex h-screen flex-col">
      <div className="flex-none p-6 bg-white border-b">
        <TrajectoryHeader />
        <TrajectoryUpload
          onLoadTrajectory={handleLoadTrajectory}
          searchParams={searchParams}
          setSearchParams={setSearchParams}
        />
      </div>
      {path && (
        <div className="flex-1 overflow-auto">
          <div className="container mx-auto py-6">
            {isLoading && (
              <Card>
                <CardContent className="py-6">
                  <div className="flex items-center justify-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Loading trajectory data...</span>
                  </div>
                </CardContent>
              </Card>
            )}
            {isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {error instanceof Error ? error.message : "Failed to load trajectory data"}
                </AlertDescription>
              </Alert>
            )}
            {trajectory && <TrajectoryViewer trajectory={trajectory} />}
          </div>
        </div>
      )}
    </div>
  );
}
