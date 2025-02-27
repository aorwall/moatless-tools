import { Card } from "@/lib/components/ui/card";
import { format } from "date-fns";
import { type Evaluation } from "../api/evaluation";

interface EvaluationStatusProps {
  evaluation: Evaluation;
}

export function EvaluationStatus({ evaluation }: EvaluationStatusProps) {
  // Calculate status counts
  const statusCounts = evaluation.instances.reduce((acc, instance) => {
    // Special handling for resolved/completed instances
    if (instance.status.toLowerCase() === 'completed') {
      if (instance.resolved === true) {
        acc['resolved'] = (acc['resolved'] || 0) + 1;
      } else if (instance.resolved === false) {
        acc['failed'] = (acc['failed'] || 0) + 1;
      } else {
        acc['completed'] = (acc['completed'] || 0) + 1;
      }
    } else {
      const status = instance.status.toLowerCase();
      acc[status] = (acc[status] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);

  // Calculate total instances
  const totalInstances = evaluation.instances.length;

  // Calculate segment widths for the progress bar
  const getSegmentWidth = (count: number) => {
    return `${(count / totalInstances) * 100}%`;
  };

  // Get status segments in order of priority
  const statusSegments = Object.entries(statusCounts)
    .map(([status, count]) => ({ status, count }))
    .filter(segment => segment.count > 0);

  const overallStatus = evaluation.status.toLowerCase();

  return (
    <Card className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h3 className="font-medium">Evaluation Status</h3>
        </div>
        <div className={`px-2 py-1 rounded-md text-xs font-medium status-bg-${overallStatus} status-text-${overallStatus}`}>
          {evaluation.status}
        </div>
      </div>

      {/* Segmented Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          {evaluation.started_at && (
            <span className="text-muted-foreground">
              Started: {format(new Date(evaluation.started_at), 'MMM d, HH:mm:ss')}
            </span>
          )}
          {evaluation.completed_at && (
            <span className="text-muted-foreground">
              Completed: {format(new Date(evaluation.completed_at), 'MMM d, HH:mm:ss')}
            </span>
          )}
        </div>
        
        <div className="h-3 rounded-full bg-gray-100 flex overflow-hidden">
          {statusSegments.map(({ status, count }) => (
            count > 0 && (
              <div
                key={status}
                className={`status-bg-${status} transition-all`}
                style={{ width: getSegmentWidth(count) }}
                title={`${status}: ${count}`}
              />
            )
          ))}
        </div>
        
        {/* Status Legend */}
        <div className="flex flex-wrap gap-3 text-xs">
          {statusSegments.map(({ status, count }) => (
            <div key={status} className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full status-bg-${status}`} />
              <span className={`font-medium status-text-${status}`}>{status}</span>
              <span className="text-muted-foreground">({count})</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
} 