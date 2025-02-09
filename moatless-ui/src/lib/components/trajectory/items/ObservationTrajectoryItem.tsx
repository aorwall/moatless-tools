export interface ObservationTimelineContent {
  message?: string;
}

export interface ObservationTrajectoryItemProps {
  content: ObservationTimelineContent;
  expandedState: boolean;
  isExpandable?: boolean;
}

export const ObservationTrajectoryItem = ({
  content,
  expandedState,
}: ObservationTrajectoryItemProps) => {
  const isExpandable = content.message
    ? content.message.length > 300 || content.message.split("\n").length > 5
    : false;

  const truncateMessage = (message?: string): string => {
    if (!message) return "";
    const lines = message.split("\n");
    if (lines.length > 5) {
      return lines.slice(0, 5).join("\n") + "\n...";
    }
    return message.length > 300 ? message.slice(0, 300) + "..." : message;
  };

  const checkMessageLength = (message: string): boolean => {
    return message.length > 300 || message.split("\n").length > 5;
  };

  const message = content.message || "No observation available";

  return (
    <div className="prose prose-sm max-w-none">
      <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
        {expandedState ? message : truncateMessage(message)}
      </p>
    </div>
  );
};
