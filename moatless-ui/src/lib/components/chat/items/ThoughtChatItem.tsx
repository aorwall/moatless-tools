import { Lightbulb } from "lucide-react";
import { ChatMessage } from "../types";

export interface ThoughtChatItemProps {
  message: ChatMessage;
}

export const ThoughtChatItem = ({ message }: ThoughtChatItemProps) => {
  return (
    <div className="flex items-start gap-2 rounded-lg bg-muted/50 px-3 py-2">
      <Lightbulb className="mt-1 h-4 w-4 text-muted-foreground" />
      <div className="min-w-0 flex-1">
        <p className="whitespace-pre-wrap break-words text-sm">
          {(message.content as { message: string }).message}
        </p>
      </div>
    </div>
  );
};
