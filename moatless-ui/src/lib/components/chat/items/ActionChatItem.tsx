import { Terminal } from "lucide-react";
import { Card } from "@/lib/components/ui/card";
import { ChatMessage } from "../types";
import { useTrajectoryStore } from "@/features/trajectory/stores/trajectoryStore";
import { cn } from "@/lib/utils";

export interface ActionChatItemProps {
  message: ChatMessage;
}

export const ActionChatItem = ({ message }: ActionChatItemProps) => {
  const setSelectedItem = useTrajectoryStore((state) => state.setSelectedItem);
  const selectedItem = useTrajectoryStore((state) => state.getSelectedItem(message.trajectoryId));

  const content = message.content as {
    properties: Record<string, any>;
    errors?: string[];
    warnings?: string[];
  };

  const isSelected = selectedItem?.itemId === message.id;

  const handleClick = () => {
    setSelectedItem(message.trajectoryId, {
      nodeId: message.nodeId,
      itemId: message.id,
      type: message.type,
      content: message.content,
    });
  };

  return (
    <Card 
      className={cn(
        "relative overflow-hidden p-3 cursor-pointer transition-colors hover:bg-muted/50",
        isSelected && "ring-2 ring-primary"
      )}
      onClick={handleClick}
    >
      <div className="flex items-center gap-2">
        <Terminal className="h-4 w-4 text-muted-foreground" />
        <div className="min-w-0 flex-1">
          <div className="space-y-1">
            <code className="text-xs font-medium rounded bg-muted px-1 py-0.5">
              {message.label}
            </code>
            {content.properties && Object.keys(content.properties).length > 0 && (
              <div className="mt-2 rounded-md bg-muted/50 p-2">
                <pre className="text-xs">
                  {JSON.stringify(content.properties, null, 2)}
                </pre>
              </div>
            )}
            {content.errors && content.errors.length > 0 && (
              <div className="mt-2 space-y-1">
                {content.errors.map((error, index) => (
                  <p key={index} className="text-xs text-destructive">
                    {error}
                  </p>
                ))}
              </div>
            )}
            {content.warnings && content.warnings.length > 0 && (
              <div className="mt-2 space-y-1">
                {content.warnings.map((warning, index) => (
                  <p key={index} className="text-xs text-warning">
                    {warning}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}; 