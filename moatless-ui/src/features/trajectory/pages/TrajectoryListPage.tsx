import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { Badge } from "@/lib/components/ui/badge";
import { Card, CardContent } from "@/lib/components/ui/card";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { useGetTrajectories } from "@/lib/hooks/useGetTrajectories";
import { formatDistanceToNow } from "date-fns";
import { AlertCircle, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";

export function TrajectoryListPage() {
  const { trajectories, isLoading, error } = useGetTrajectories();

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="py-6">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Loading runs...</span>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load runs"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 h-[calc(100vh-56px)] flex flex-col">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Runs</h1>
      </div>

      <ScrollArea className="flex-1 w-full">
        <div className="grid gap-4 pr-4">
          {trajectories.map((trajectory) => (
            <Link
              key={`${trajectory.project_id}-${trajectory.trajectory_id}`}
              to={`/trajectories/${trajectory.project_id}/${trajectory.trajectory_id}`}
            >
              <Card className="hover:bg-muted/50 transition-colors">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">
                          {trajectory.project_id}/{trajectory.trajectory_id}
                        </span>
                        <Badge variant={getStatusVariant(trajectory.status)}>
                          {trajectory.status}
                        </Badge>
                      </div>
                      <div className="flex flex-col text-sm text-muted-foreground">
                        <span>
                          Started{" "}
                          {formatDistanceToNow(new Date(trajectory.started_at))}{" "}
                          ago
                        </span>
                        {trajectory.finished_at && (
                          <span>
                            Finished{" "}
                            {formatDistanceToNow(
                              new Date(trajectory.finished_at),
                            )}{" "}
                            ago
                          </span>
                        )}
                      </div>
                    </div>
                    {trajectory.error && (
                      <div className="text-sm text-destructive max-w-[300px] truncate">
                        {trajectory.error}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

function getStatusVariant(
  status: string,
): "default" | "secondary" | "destructive" {
  switch (status) {
    case "completed":
      return "default";
    case "running":
      return "secondary";
    case "error":
      return "destructive";
    default:
      return "default";
  }
}
