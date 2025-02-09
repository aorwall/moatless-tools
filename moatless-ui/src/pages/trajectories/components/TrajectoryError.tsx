import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Trajectory } from "@/lib/types/trajectory";

interface TrajectoryErrorProps {
  trajectory: Trajectory;
}

export function TrajectoryError({ trajectory }: TrajectoryErrorProps) {
  if (!trajectory?.system_status.error) return null;

  return (
    <ScrollArea className="flex-1 w-full">
      <div className="p-6 space-y-4">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Execution Error</AlertTitle>
          <AlertDescription>
            {trajectory.system_status.error}
          </AlertDescription>
        </Alert>

        {trajectory.system_status.error_trace && (
          <div className="space-y-2">
            <h3 className="text-sm font-medium">Error Trace</h3>
            <pre className="whitespace-pre-wrap break-words rounded-md bg-destructive/10 p-4 text-sm text-destructive font-mono">
              {trajectory.system_status.error_trace}
            </pre>
          </div>
        )}
      </div>
    </ScrollArea>
  );
} 