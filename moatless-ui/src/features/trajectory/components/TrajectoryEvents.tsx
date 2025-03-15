import { useGetTrajectoryEvents } from "@/features/trajectory/hooks/useGetTrajectoryEvents";
import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import { Skeleton } from "@/lib/components/ui/skeleton.tsx";
import { TrajectoryEvent } from "@/lib/types/trajectory.ts";
import { format } from "date-fns";
import { AlertCircle } from "lucide-react";

interface TrajectoryEventsProps {
  projectId: string;
  trajectoryId: string;
  status?: string;
}

export function TrajectoryEvents({
  projectId,
  trajectoryId,
  status,
}: TrajectoryEventsProps) {
  const { data, isLoading, error } = useGetTrajectoryEvents(
    projectId,
    trajectoryId,
    status
  );


  if (isLoading) {
    return (
      <div className="flex flex-col h-full p-4 space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="flex-1 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-full p-4 text-destructive">
        <div className="flex items-center gap-2">
          <AlertCircle className="h-5 w-5" />
          <p>Error loading events: {error.toString()}</p>
        </div>
      </div>
    );
  }

  const events = data || [];

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-4 text-muted-foreground">
        <p>No events</p>
      </div>
    );
  }

  // Sort events by timestamp in descending order (newest first)
  const sortedEvents = [...events].sort((a, b) => {
    return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
  });

  // Function to get badge style for scope
  const getScopeBadgeStyle = (scope: string) => {
    switch (scope) {
      case 'flow':
        return 'bg-blue-500/20 text-blue-700 dark:text-blue-400';
      case 'evaluation':
        return 'bg-purple-500/20 text-purple-700 dark:text-purple-400';
      case 'node':
        return 'bg-amber-500/20 text-amber-700 dark:text-amber-400';
      case 'action':
        return 'bg-sky-500/20 text-sky-700 dark:text-sky-400';
      case 'agent':
        return 'bg-indigo-500/20 text-indigo-700 dark:text-indigo-400';
      default:
        return 'bg-muted/50 text-muted-foreground';
    }
  };

  // Function to get badge style for event type
  const getEventTypeBadgeStyle = (eventType: string, scope: string) => {
    if (scope === 'flow' && eventType === 'completed') {
      return 'bg-green-500/20 text-green-700 dark:text-green-400';
    } else if (scope === 'flow' && eventType === 'started') {
      return 'bg-blue-500/30 text-blue-700 dark:text-blue-400';
    } else if (eventType === 'error') {
      return 'bg-red-500/20 text-red-700 dark:text-red-400';
    } else {
      return 'bg-muted text-muted-foreground';
    }
  };

  return (
    <ScrollArea className="h-full w-full font-mono">
      <div className="p-0">
        <table className="w-full text-xs border-collapse">
          <tbody>
            {sortedEvents.map((event, index) => {
              const date = new Date(event.timestamp);
              const time = format(date, 'HH:mm:ss');

              // Group events by type to reduce visual clutter
              const isRepeatedEvent = index > 0 &&
                event.event_type === sortedEvents[index - 1].event_type &&
                event.scope === sortedEvents[index - 1].scope;

              if (isRepeatedEvent) {
                return null; // Skip repeated events of the same type
              }

              // Extract all relevant properties
              const {
                scope,
                event_type,
                node_id,
                agent_id,
                action_name,
                data
              } = event;

              // Access potentially undefined properties safely
              const child_node_id = (event as any).child_node_id;
              const previous_node_id = (event as any).previous_node_id;
              const action_params = (event as any).action_params;

              // Build a list of properties to display (excluding scope and event_type)
              const properties = [];
              if (node_id !== undefined) properties.push(`node=${String(node_id)}`);
              if (child_node_id !== undefined) properties.push(`child=${String(child_node_id)}`);
              if (previous_node_id !== undefined) properties.push(`prev=${String(previous_node_id)}`);
              if (agent_id) properties.push(`agent=${agent_id}`);
              if (action_name) properties.push(`action=${action_name}`);

              const scopeBadgeStyle = getScopeBadgeStyle(scope || '');
              const eventTypeBadgeStyle = getEventTypeBadgeStyle(event_type || '', scope || '');

              return (
                <tr
                  key={index}
                  className={`${index % 2 === 0 ? 'bg-muted/30' : 'bg-transparent'} hover:bg-muted/50`}
                >
                  <td className="py-1 px-2 text-muted-foreground w-20 border-r border-border whitespace-nowrap">
                    {time}
                  </td>
                  <td className="py-1 px-2">
                    <div className="flex flex-wrap gap-1.5 items-center">
                      {scope && (
                        <span className={`inline-block px-1.5 py-0.5 rounded font-medium ${scopeBadgeStyle}`}>
                          {scope}
                        </span>
                      )}
                      {event_type && (
                        <span className={`inline-block px-1.5 py-0.5 rounded font-medium ${eventTypeBadgeStyle}`}>
                          {event_type}
                        </span>
                      )}
                      {properties.map((prop, i) => (
                        <span key={i} className="inline-block bg-muted/50 px-1.5 py-0.5 rounded">
                          {prop}
                        </span>
                      ))}
                    </div>
                    {data && Object.keys(data).length > 0 && (
                      <pre className="mt-1 text-xs font-mono bg-muted/30 p-1.5 rounded overflow-x-auto max-h-24 overflow-y-auto">
                        {JSON.stringify(data, null, 2)}
                      </pre>
                    )}
                    {action_params && (
                      <pre className="mt-1 text-xs font-mono bg-muted/30 p-1.5 rounded overflow-x-auto max-h-24 overflow-y-auto">
                        {JSON.stringify(action_params, null, 2)}
                      </pre>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </ScrollArea>
  );
}
