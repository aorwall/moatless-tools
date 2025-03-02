import { cn } from "@/lib/utils";
import { ChatMessage } from "../types";

export interface MessageChatItemProps {
  message: ChatMessage;
  isUser: boolean;
  isExpanded: boolean;
  onExpandChange: () => void;
}

export const MessageChatItem = ({
  message,
  isUser,
  isExpanded,
  onExpandChange,
}: MessageChatItemProps) => {
  const isExpandable =
    (message.content as { message: string }).message.length > 300 ||
    (message.content as { message: string }).message.split("\n").length > 5;

  const truncateMessage = (text: string) => {
    if (text.length > 300) {
      return text.slice(0, 300) + "...";
    }
    const lines = text.split("\n");
    if (lines.length > 5) {
      return lines.slice(0, 5).join("\n") + "\n...";
    }
    return text;
  };

  return (
    <div
      className={cn(
        "relative flex min-w-0 flex-col gap-1 rounded-lg px-3 py-2",
        isUser
          ? "bg-primary text-primary-foreground"
          : "bg-muted text-foreground",
      )}
    >
      <p className="whitespace-pre-wrap break-words text-sm">
        {isExpandable && !isExpanded
          ? truncateMessage((message.content as { message: string }).message)
          : (message.content as { message: string }).message}
      </p>

      {isExpandable && (
        <button
          className="absolute -bottom-7 left-1/2 -translate-x-1/2 transform text-muted-foreground hover:text-foreground"
          onClick={onExpandChange}
        >
          {isExpanded ? "Show less" : "Show more"}
        </button>
      )}

      {message.nodeId !== undefined && (
        <span className="absolute -bottom-5 right-2 text-[10px] text-muted-foreground">
          Step {message.nodeId}
        </span>
      )}
    </div>
  );
};
