
export interface ErrorTimelineContent {
  error: string;
}

export interface ErrorTrajectoryItemProps {
  content: ErrorTimelineContent;
}

export const ErrorTrajectoryItem = ({ content }: ErrorTrajectoryItemProps) => {
  const firstLine = content.error.split("\n")[0];
  const remainingLines = content.error.split("\n").length - 1;

  return (
    <div className="text-xs text-red-600">
      {firstLine}
      {remainingLines > 0 && (
        <span className="text-gray-400 ml-2">
          and {remainingLines} more lines
        </span>
      )}
    </div>
  );
};
