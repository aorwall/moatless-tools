export interface ObservationTimelineContent {
  message?: string;
}

export interface ObservationTrajectoryItemProps {
  content: ObservationTimelineContent;
}

export const ObservationTrajectoryItem = ({
  content,
}: ObservationTrajectoryItemProps) => {
  const truncateMessage = (message?: string): string => {
    if (!message) return "";
    const lines = message.split("\n");
    if (lines.length > 5) {
      return lines.slice(0, 5).join("\n") + "\n...";
    }
    return message.length > 300 ? message.slice(0, 300) + "..." : message;
  };

  const message = content.message || "No observation available";

  return (
    <div className="prose prose-sm max-w-none">
      <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
        {truncateMessage(message)}
      </p>
    </div>
  );
};
