import { Badge } from "@/lib/components/ui/badge";
import { format } from "date-fns";
import { type Evaluation } from "../api/evaluation";
import { Link } from "react-router-dom";

interface EvaluationTimelineProps {
  evaluation: Evaluation;
  getStatusColor: (status: string) => "default" | "secondary" | "destructive" | "outline";
}

const progressAnimation = `
  @keyframes progress {
    0% {
      background-position: 0% 50%;
    }
    100% {
      background-position: 100% 50%;
    }
  }
`;

interface TimelineBarProps {
  instance: Evaluation["instances"][number];
  isFirstSegment: boolean;
  left: number;
  width: number;
  children?: React.ReactNode;
}

function TimelineBar({ instance, isFirstSegment, left, width, children }: TimelineBarProps) {
  const baseStyles = "absolute inset-y-0 transition-all group-hover:scale-y-110";
  
  const getBarStyles = () => {
    if (isFirstSegment && !instance.completed_at) {
      // Running instance (first segment)
      return `${baseStyles} bg-gradient-to-r from-primary/20 via-primary/40 to-primary/20 bg-[length:200%_100%] animate-shimmer`;
    } else if (isFirstSegment) {
      // Completed first segment
      return `${baseStyles} bg-primary/30`;
    } else {
      // Evaluation segment
      return `${baseStyles} bg-primary/60`;
    }
  };

  return (
    <div
      className={getBarStyles()}
      style={{
        left: `${left}%`,
        width: `${width}%`,
      }}
    >
      {children}
    </div>
  );
}

export function EvaluationTimeline({ evaluation, getStatusColor }: EvaluationTimelineProps) {
  const now = new Date();
  const instances = evaluation.instances;
  
  // Add validation for timestamps
  const isValidDate = (timestamp: string | null | undefined): boolean => {
    if (!timestamp) return false;
    const date = new Date(timestamp);
    return date instanceof Date && !isNaN(date.getTime());
  };

  // Filter out not started instances and sort by started_at timestamp
  const sortedInstances = [...instances]
    .filter(instance => {
      // Include instances that have a valid started_at timestamp or are currently running
      return isValidDate(instance.started_at) || instance.status === "running";
    })
    .sort((a, b) => {
      // For running instances without started_at, use current time
      const aTime = a.started_at && isValidDate(a.started_at) ? new Date(a.started_at).getTime() : now.getTime();
      const bTime = b.started_at && isValidDate(b.started_at) ? new Date(b.started_at).getTime() : now.getTime();
      return aTime - bTime;
    });

  
  // Only continue if we have instances to show
  if (sortedInstances.length === 0) {
    return <div>No active instances</div>;
  }
  
  // Update the getEndTime function with validation
  const getEndTime = (instance: Evaluation["instances"][number]) => {
    if (instance.completed_at && isValidDate(instance.completed_at)) return instance.completed_at;
    // For running instances, end 5 seconds before current time
    return new Date(now.getTime() - 5000).toISOString();
  };

  // Update the timestamps calculation with validation
  const timestamps = sortedInstances.flatMap(instance => [
    instance.started_at,
    getEndTime(instance),
    instance.evaluated_at
  ].filter(timestamp => isValidDate(timestamp)) as string[]);
  
  if (timestamps.length === 0) {
    return <div>No valid timestamps available</div>;
  }

  const startTime = new Date(Math.min(...timestamps.map(t => new Date(t).getTime())));
  const endTime = new Date(Math.max(...timestamps.map(t => new Date(t).getTime())));
  const totalDuration = endTime.getTime() - startTime.getTime();

  // Update getPositionPercentage with validation
  const getPositionPercentage = (timestamp: string | null) => {
    if (!isValidDate(timestamp)) return 0;
    const time = new Date(timestamp).getTime();
    return ((time - startTime.getTime()) / totalDuration) * 100;
  };

  return (
    <div className="space-y-2">
      <style>{progressAnimation}</style>
      
      <div className="flex gap-4">
        {/* Fixed-width column for instance IDs */}
        <div className="w-32 flex-none">
          {/* Empty space to align with timeline */}
        </div>
        
        {/* Timeline header with timestamps */}
        <div className="flex-1">
          <div className="flex justify-between text-xs text-muted-foreground mb-4">
            <span>{format(startTime, 'HH:mm:ss')}</span>
            <span>{format(endTime, 'HH:mm:ss')}</span>
          </div>
        </div>
      </div>
      
      {sortedInstances.map((instance) => (
        <div key={instance.instance_id} className="flex gap-4">
          {/* Instance ID column */}
          <div className="w-32 flex-none">
            <Link
              to={`/swebench/evaluation/${evaluation.evaluation_name}/${instance.instance_id}`}
              className="text-xs font-mono text-right block py-1 hover:text-primary transition-colors"
            >
              {instance.instance_id}
            </Link>
          </div>
          
          {/* Timeline column */}
          <div className="flex-1 relative h-6 group">
            <div className="absolute inset-y-0 left-0 w-full bg-muted/20 rounded-full" />
            
            {instance.started_at && (
              <TimelineBar
                instance={instance}
                isFirstSegment={true}
                left={getPositionPercentage(instance.started_at)}
                width={getPositionPercentage(getEndTime(instance)) - getPositionPercentage(instance.started_at)}
              >
                {!instance.completed_at && (
                  <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-[2px]">
                    <Badge 
                      variant={getStatusColor(instance.status)} 
                      className="text-[10px] px-1.5 py-0"
                    >
                      {instance.status}
                    </Badge>
                  </div>
                )}
              </TimelineBar>
            )}
            
            {instance.completed_at && instance.evaluated_at && (
              <TimelineBar
                instance={instance}
                isFirstSegment={false}
                left={getPositionPercentage(instance.completed_at)}
                width={getPositionPercentage(instance.evaluated_at) - getPositionPercentage(instance.completed_at)}
              >
                <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-[2px]">
                  <Badge 
                    variant={instance.resolved ? "default" : "destructive"} 
                    className="text-[10px] px-1.5 py-0"
                  >
                    {instance.resolved ? "✓" : "✗"}
                  </Badge>
                </div>
              </TimelineBar>
            )}
          </div>
        </div>
      ))}
      
      {/* Legend */}
      <div className="flex gap-4">
        <div className="w-32 flex-none" />
        <div className="flex-1">
          <div className="flex text-xs text-muted-foreground mt-2 gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-primary/30 rounded" />
              <span>Running</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-primary/60 rounded" />
              <span>Evaluating</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 