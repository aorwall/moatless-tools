import { Badge } from '@/lib/components/ui/badge';
import { AlertCircle, CheckCircle2, Loader2, History, Zap, Coins, FileText } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface RunStatusProps {
  status: {
    status: string;
    started_at: string;
    finished_at?: string;
    restart_count: number;
    error?: string;
  };
  trajectory?: {
    iterations?: number;
    completionCost?: number;
    totalTokens?: number;
    promptTokens?: number;
    completionTokens?: number;
    cachedTokens?: number;
    flags: string[];
    failedActions: number;
    duplicatedActions: number;
  };
}

export function RunStatusDisplay({ status, trajectory }: RunStatusProps) {
  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'error':
        return <AlertCircle className="h-5 w-5 text-destructive" />;
      case 'finished':
        return <CheckCircle2 className="h-5 w-5 text-success" />;
      case 'running':
        return <Loader2 className="h-5 w-5 animate-spin" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">Current Status</span>
        <Badge 
          variant={status.status === 'error' ? 'destructive' : 'default'}
          className="flex items-center gap-1"
        >
          {getStatusIcon(status.status)}
          {status.status}
        </Badge>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span>Started</span>
          <span>{formatDistanceToNow(new Date(status.started_at))} ago</span>
        </div>
        {status.finished_at && (
          <div className="flex justify-between text-sm">
            <span>Finished</span>
            <span>{formatDistanceToNow(new Date(status.finished_at))} ago</span>
          </div>
        )}
      </div>

      {trajectory && (
        <>
          <div className="border-t pt-4">
            <h3 className="text-sm font-medium mb-3">Execution Stats</h3>
            <div className="grid grid-cols-2 gap-3">
              {trajectory.iterations !== undefined && (
                <div className="flex items-center gap-2 text-sm">
                  <Zap className="h-4 w-4 text-muted-foreground" />
                  <span>{trajectory.iterations} iterations</span>
                </div>
              )}
              {trajectory.completionCost !== undefined && (
                <div className="flex items-center gap-2 text-sm">
                  <Coins className="h-4 w-4 text-muted-foreground" />
                  <span>${trajectory.completionCost.toFixed(4)}</span>
                </div>
              )}
            </div>
          </div>

          <div className="border-t pt-4">
            <h3 className="text-sm font-medium mb-3">Token Usage</h3>
            <div className="space-y-2">
              {trajectory.promptTokens !== undefined && (
                <div className="flex justify-between text-sm">
                  <span>Prompt</span>
                  <span>{trajectory.promptTokens.toLocaleString()}</span>
                </div>
              )}
              {trajectory.completionTokens !== undefined && (
                <div className="flex justify-between text-sm">
                  <span>Completion</span>
                  <span>{trajectory.completionTokens.toLocaleString()}</span>
                </div>
              )}
              {trajectory.cachedTokens !== undefined && (
                <div className="flex justify-between text-sm">
                  <span>Cached</span>
                  <span>{trajectory.cachedTokens.toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>

          {(trajectory.failedActions > 0 || trajectory.duplicatedActions > 0) && (
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium mb-3">Issues</h3>
              <div className="space-y-2">
                {trajectory.failedActions > 0 && (
                  <div className="flex justify-between text-sm text-destructive">
                    <span>Failed Actions</span>
                    <span>{trajectory.failedActions}</span>
                  </div>
                )}
                {trajectory.duplicatedActions > 0 && (
                  <div className="flex justify-between text-sm text-warning">
                    <span>Duplicated Actions</span>
                    <span>{trajectory.duplicatedActions}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {trajectory.flags.length > 0 && (
            <div className="border-t pt-4">
              <h3 className="text-sm font-medium mb-3">Flags</h3>
              <div className="flex flex-wrap gap-2">
                {trajectory.flags.map((flag, index) => (
                  <Badge key={index} variant="secondary">
                    {flag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {status.error && (
        <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
          {status.error}
        </div>
      )}
    </div>
  );
}

export { RunStatusDisplay as RunStatus }; 