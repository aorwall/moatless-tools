import { Package } from "lucide-react";
import { Card } from "@/lib/components/ui/card";
import { ChatMessage } from "../types";
import { useTrajectoryStore } from "@/features/trajectory/stores/trajectoryStore";
import { cn } from "@/lib/utils";

export interface ArtifactChatItemProps {
  message: ChatMessage;
}

export const ArtifactChatItem = ({ message }: ArtifactChatItemProps) => {
  const setSelectedItem = useTrajectoryStore((state) => state.setSelectedItem);
  const selectedItem = useTrajectoryStore((state) => state.getSelectedItem(message.trajectoryId));

  const content = message.content as {
    artifact_id: string;
    artifact_type: string;
    change_type: string;
    diff_details: any;
    actor: string;
  };

  const isSelected = selectedItem?.itemId === message.id;
  const isUser = content.actor === "user";

  const handleClick = () => {
    setSelectedItem(message.trajectoryId, {
      nodeId: message.nodeId,
      itemId: message.id,
      type: message.type,
      content: message.content,
    });
  };

  return (
    <div className={cn("flex w-full", isUser && "justify-end")}>
      <Card 
        className={cn(
          "relative overflow-hidden p-3 cursor-pointer transition-colors hover:bg-muted/50",
          isSelected && "ring-2 ring-primary",
          "max-w-[85%] border-border/50 bg-background/80"
        )}
        onClick={handleClick}
      >
        <div className={cn("flex items-center gap-2", isUser && "flex-row-reverse")}>
          <Package className="h-4 w-4 text-muted-foreground" />
          <div className="min-w-0 flex-1">
            <div className="space-y-1">
              <p className="text-xs font-medium">{content.artifact_id}</p>
              <div className={cn("flex flex-wrap gap-2", isUser && "justify-end")}>
                <span className="text-xs text-muted-foreground">
                  Type: {content.artifact_type}
                </span>
                <span className="text-xs text-muted-foreground">
                  Change: {content.change_type}
                </span>
                <span className="text-xs text-muted-foreground">
                  Actor: {content.actor}
                </span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}; 