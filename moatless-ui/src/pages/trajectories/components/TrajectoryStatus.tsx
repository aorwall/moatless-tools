import { Badge } from "@/lib/components/ui/badge";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Zap,
  Coins,
  Clock,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { Trajectory } from "@/lib/types/trajectory";

interface TrajectoryStatusProps {
  trajectory: Trajectory
}

export function TrajectoryStatus({ trajectory }: TrajectoryStatusProps) {
  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case "finished":
        return <CheckCircle2 className="h-4 w-4 text-success" />;
      case "running":
        return <Loader2 className="h-4 w-4 animate-spin" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-2 p-3">
      {/* Status and Stats */}
      <div className="flex items-center justify-between">
        <Badge
          variant={trajectory?.status === "error" ? "destructive" : "default"}
          className="flex items-center gap-1"
        >
          {getStatusIcon(trajectory?.status)}
          {trajectory?.status}
        </Badge>
        <div className="flex gap-3 text-xs">
          {trajectory.nodes.length !== undefined && (
            <div className="flex items-center gap-1">
              <Zap className="h-3 w-3 text-muted-foreground" />
              <span>{trajectory.nodes.length}</span>
            </div>
          )}
          {trajectory.completionCost !== undefined && (
            <div className="flex items-center gap-1">
              <Coins className="h-3 w-3 text-muted-foreground" />
              <span>${trajectory.completionCost.toFixed(4)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Timing Info */}
      <div className="grid grid-cols-2 gap-x-2 text-xs">
        <div className="flex items-center gap-1 text-muted-foreground">
          <Clock className="h-3 w-3" /> Started: {formatDistanceToNow(new Date(trajectory?.system_status.started_at))} ago
        </div>
        {trajectory?.system_status.finished_at && (
          <div className="flex items-center gap-1 text-muted-foreground">
            <Clock className="h-3 w-3" /> Finished: {formatDistanceToNow(new Date(trajectory?.system_status.finished_at))} ago
          </div>
        )}
      </div>

      {/* Token Usage and Issues */}
      <div className="grid grid-cols-2 gap-x-4 text-xs">
        <div className="space-y-1">
          <div className="text-muted-foreground font-medium">Token Usage</div>
          {trajectory.promptTokens !== undefined && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Prompt:</span>
              <span>{trajectory.promptTokens.toLocaleString()}</span>
            </div>
          )}
          {trajectory.completionTokens !== undefined && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Completion:</span>
              <span>{trajectory.completionTokens.toLocaleString()}</span>
            </div>
          )}
          {trajectory.cachedTokens !== undefined && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Cached:</span>
              <span>{trajectory.cachedTokens.toLocaleString()}</span>
            </div>
          )}
        </div>

        {((trajectory.failedActions !== undefined && trajectory.failedActions > 0) ||
          (trajectory.duplicatedActions !== undefined && trajectory.duplicatedActions > 0)) && (
          <div className="space-y-1">
            <div className="text-muted-foreground font-medium">Issues</div>
            {trajectory.failedActions !== undefined && trajectory.failedActions > 0 && (
              <div className="flex justify-between text-destructive">
                <span>Failed:</span>
                <span>{trajectory.failedActions}</span>
              </div>
            )}
            {trajectory.duplicatedActions !== undefined && trajectory.duplicatedActions > 0 && (
              <div className="flex justify-between text-warning">
                <span>Duplicated:</span>
                <span>{trajectory.duplicatedActions}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Flags */}
      {trajectory.flags !== undefined && trajectory.flags.length > 0 && (
        <div className="text-xs">
          <div className="text-muted-foreground font-medium mb-1">Flags</div>
          <div className="flex flex-wrap gap-1">
            {trajectory.flags.map((flag, index) => (
              <Badge key={index} variant="secondary" className="text-xs">
                {flag}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

