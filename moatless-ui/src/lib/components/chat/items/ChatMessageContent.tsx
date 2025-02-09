import { ChatMessage } from "../types";

interface ChatMessageContentProps {
  message: ChatMessage;
}

export function ChatMessageContent({ message }: ChatMessageContentProps) {
  switch (message.type) {
    case "user_message":
    case "assistant_message":
    case "thought":
      return (
        <p className="whitespace-pre-wrap break-words text-sm">
          {message.content.message}
        </p>
      );
    case "action":
      return (
        <div className="space-y-1">
          <p className="text-xs font-medium">{message.content.name}</p>
          {message.content.shortSummary && (
            <p className="font-mono text-xs">{message.content.shortSummary}</p>
          )}
          {message.content.properties && Object.keys(message.content.properties).length > 0 && (
            <div className="mt-2 rounded-md bg-muted/50 p-2">
              <pre className="text-xs">
                {JSON.stringify(message.content.properties, null, 2)}
              </pre>
            </div>
          )}
          {message.content.errors && message.content.errors.length > 0 && (
            <div className="mt-2 space-y-1">
              {message.content.errors.map((error: string, index: number) => (
                <p key={index} className="text-xs text-destructive">
                  {error}
                </p>
              ))}
            </div>
          )}
          {message.content.warnings && message.content.warnings.length > 0 && (
            <div className="mt-2 space-y-1">
              {message.content.warnings.map((warning: string, index: number) => (
                <p key={index} className="text-xs text-warning">
                  {warning}
                </p>
              ))}
            </div>
          )}
        </div>
      );
    case "artifact":
      return (
        <div className="space-y-1">
          <p className="text-xs font-medium">{message.content.name}</p>
          {message.content.description && (
            <p className="text-xs text-muted-foreground">
              {message.content.description}
            </p>
          )}
          {message.content.type && (
            <p className="text-xs text-muted-foreground">
              Type: {message.content.type}
            </p>
          )}
        </div>
      );
    default:
      return null;
  }
} 