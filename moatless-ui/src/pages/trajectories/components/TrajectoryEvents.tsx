import { RunEvent } from "@/lib/types/run";
import {
  AlertCircle,
  Info,
  MessageSquare,
  Bot,
  Terminal,
  Loader2,
  GitBranch,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { TrajectoryEvent } from "@/lib/types/trajectory";
interface TrajectoryEventsProps {
  events: TrajectoryEvent[];
  className?: string;
}

export function TrajectoryEvents({ events, className }: TrajectoryEventsProps) {
  // just sort in reverse order
  const reversedEvents = [...events].reverse();

  const getEventIcon = (event: TrajectoryEvent) => {
    if (event.event_type.includes("error")) {
      return <AlertCircle className="h-4 w-4 text-destructive" />;
    } else if (event.scope === "agent" || event.scope === "action") {
      return <Bot className="h-4 w-4 text-primary" />;
    } else if (event.scope === "flow") {
      return <GitBranch className="h-4 w-4 text-blue-500" />;
    } else {
        return <Info className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const formatEventType = (type: string) => {
    return type
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground">
        No events yet
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      {reversedEvents.map((event, i) => (
        <div
          key={`${event.timestamp}-${i}`}
          className={cn(
            "rounded-md border bg-card p-3 text-sm shadow-sm",
            event.event_type.includes("error") && "bg-destructive/10 border-destructive/20"
          )}
        >
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              {getEventIcon(event)}
              <span className="font-medium">
                {formatEventType(event.event_type)}
              </span>
              {event.node_id !== undefined && (
                <span className="text-xs text-muted-foreground">
                  Node {event.node_id}
                </span>
              )}
            </div>
            <span className="text-xs text-muted-foreground">
              {formatDistanceToNow(new Date(event.timestamp))} ago
            </span>
          </div>
          {event.action_name && (
            <p className="text-xs text-muted-foreground mt-1">
              Action: {event.action_name}
            </p>
          )}
          {event.data && Object.entries(event.data).length > 0 && (
            <div className="mt-2 space-y-1">
              {Object.entries(event.data).map(([key, value]) => (
                <p key={key} className="text-xs text-muted-foreground">
                  {key}: {JSON.stringify(value)}
                </p>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
