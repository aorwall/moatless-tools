import { RunEvent } from '@/lib/types/run';
import { AlertCircle, Info, MessageSquare, Bot, Terminal, Loader2 } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { ScrollArea } from '@/lib/components/ui/scroll-area';
import { cn } from '@/lib/utils';

interface RunEventsProps {
  events: RunEvent[];
  className?: string;
}

export function RunEvents({ events, className }: RunEventsProps) {
  // just sort in reverse order
  const reversedEvents = [...events].reverse();

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      case 'agent_message':
        return <Bot className="h-4 w-4 text-primary" />;
      case 'system_message':
        return <Terminal className="h-4 w-4 text-muted-foreground" />;
      case 'user_message':
        return <MessageSquare className="h-4 w-4 text-blue-500" />;
      case 'agent_action_created':
        return <Bot className="h-4 w-4 text-primary" />;
      case 'agent_action_executed':
        return <Loader2 className="h-4 w-4 text-green-500" />;
      default:
        return <Info className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const formatEventType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
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
          className="rounded-md border bg-card p-3 text-sm shadow-sm"
        >
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              {getEventIcon(event.event_type)}
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
        </div>
      ))}
    </div>
  );
} 