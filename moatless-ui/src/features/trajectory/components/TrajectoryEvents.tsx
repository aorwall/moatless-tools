import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import { TrajectoryEvent } from "@/lib/types/trajectory.ts";
import { formatDistanceToNow } from "date-fns";

interface TrajectoryEventsProps {
  events: TrajectoryEvent[];
}

export function TrajectoryEvents({ events }: TrajectoryEventsProps) {
  if (!events || events.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-4 text-muted-foreground">
        <p>No events</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full w-full">
      <div className="p-4">
        {events.map((event, index) => {
          const date = new Date(event.timestamp);
          const timeAgo = formatDistanceToNow(date, { addSuffix: true });

          return (
            <div key={index} className="mb-4 last:mb-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-foreground">
                  {event.event_type}
                </span>
                <span className="text-xs text-muted-foreground">
                  {timeAgo}
                </span>
              </div>
              <div className="flex gap-2 items-start mt-1">
                <div className="w-1 h-1 rounded-full bg-muted-foreground mt-1.5"></div>
                <div>
                  <div className="text-sm">
                    {event.scope && <span className="text-muted-foreground">{event.scope}</span>}
                    {event.node_id !== undefined && <span className="ml-1 text-muted-foreground">Node: {event.node_id}</span>}
                    {event.action_name && <span className="ml-1 text-muted-foreground">Action: {event.action_name}</span>}
                  </div>
                  {event.data && (
                    <pre className="mt-1 text-xs font-mono bg-muted p-2 rounded overflow-x-auto">
                      {JSON.stringify(event.data, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </ScrollArea>
  );
}
