import React, { useEffect } from "react";

export interface ErrorTimelineContent {
  error: string;
}

export interface ErrorTrajectoryItemProps {
  content: ErrorTimelineContent;
  expandedState: boolean;
  isExpandable?: boolean;
}

export const ErrorTrajectoryItem = ({
  content,
  expandedState,
}: ErrorTrajectoryItemProps) => {
  const isExpandable = content.error.split("\n").length > 1;

  if (expandedState) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <pre className="text-sm text-red-700 whitespace-pre-wrap font-mono">
          {content.error}
        </pre>
      </div>
    );
  }

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
