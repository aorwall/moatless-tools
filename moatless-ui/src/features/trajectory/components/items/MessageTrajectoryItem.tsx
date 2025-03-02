export interface MessageTimelineContent {
  message: string;
}

export interface MessageTrajectoryItemProps {
  content: MessageTimelineContent;
  type: "user_message" | "assistant_message" | "thought";
}

export const MessageTrajectoryItem = ({
  content,
  type,
}: MessageTrajectoryItemProps) => {
  const truncateMessage = (message?: string): string => {
    if (!message) return "";
    return message.length > 200 ? message.slice(0, 200) + "..." : message;
  };

  return (
    <div className="prose prose-sm max-w-none">
      <p className="whitespace-pre-wrap text-xs leading-relaxed text-gray-700">
        {truncateMessage(content.message)}
      </p>
    </div>
  );
};
