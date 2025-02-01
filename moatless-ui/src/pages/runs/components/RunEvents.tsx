import { RunEvent } from '@/lib/types/run';
import { AlertCircle, Info, MessageSquare, Bot, Terminal } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface RunEventsProps {
  events: RunEvent[];
}

export function RunEvents({ events }: RunEventsProps) {
  // desc sort events by timestamp
  const sortedEvents = events.sort((a, b) => b.timestamp - a.timestamp);

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case 'agent_message':
        return <Bot className="h-4 w-4" />;
      case 'system_message':
        return <Terminal className="h-4 w-4" />;
      case 'user_message':
        return <MessageSquare className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-2 p-4">
      {sortedEvents.map((event, i) => (
        <div 
          key={i} 
          className="rounded-md bg-muted p-3 text-sm"
        >
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              {getEventIcon(event.event_type)}
              <span className="font-medium">
                {event.event_type.replace(/_/g, ' ')}
              </span>
            </div>
            <span className="text-xs text-muted-foreground">
              {formatDistanceToNow(new Date(event.timestamp))} ago
            </span>
          </div>
          {event.data.message && (
            <p className="text-sm text-muted-foreground mt-1">
              {event.data.message}
            </p>
          )}
        </div>
      ))}
    </div>
  );
} 