import { Link } from "react-router-dom";
import { Card, CardContent } from "@/lib/components/ui/card";
import { Loader2, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/lib/components/ui/alert";
import { Badge } from "@/lib/components/ui/badge";
import { formatDistanceToNow } from "date-fns";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { useGetTrajectories } from "@/lib/hooks/useGetTrajectories";

export function TrajectoriesPage() {
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
          {trajectories.map((run) => (
            <Link key={run.trajectory_id} to={`/trajectories/${run.trajectory_id}`}>
              <Card className="hover:bg-muted/50 transition-colors">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{run.trajectory_id}</span>
                        <Badge variant={getStatusVariant(run.status)}>
                          {run.status}
                        </Badge>
                      </div>
                      <div className="flex flex-col text-sm text-muted-foreground">
                        <span>
                          Started {formatDistanceToNow(new Date(run.started_at))} ago
                        </span>
                        {run.finished_at && (
                          <span>
                            Finished {formatDistanceToNow(new Date(run.finished_at))} ago
                          </span>
                        )}
                      </div>
                    </div>
                    {run.error && (
                      <div className="text-sm text-destructive max-w-[300px] truncate">
                        {run.error}
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

function getStatusVariant(status: string): "default" | "secondary" | "destructive" {
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