import { Badge } from "@/lib/components/ui/badge.tsx";
import { FC } from "react";

export interface ArtifactTimelineContent {
  artifact_id: string;
  artifact_type: string;
  change_type: "added" | "updated" | "removed";
  diff_details?: string;
  actor: "user" | "assistant";
}

export interface ArtifactTrajectoryItemProps {
  content: ArtifactTimelineContent;
}

export const ArtifactTrajectoryItem: FC<ArtifactTrajectoryItemProps> = ({
  content,
}) => {
  const getChangeTypeBadge = (type: string) => {
    switch (type) {
      case "added":
        return <Badge variant="success">Added</Badge>;
      case "updated":
        return <Badge variant="default">Updated</Badge>;
      case "removed":
        return <Badge variant="destructive">Removed</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-700">
          {content.artifact_type}
        </span>
        {getChangeTypeBadge(content.change_type)}
        <span className="text-xs text-gray-500">by {content.actor}</span>
      </div>

      <div className="rounded-md bg-gray-50 p-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <span className="font-medium">ID:</span>
            <span className="font-mono">{content.artifact_id}</span>
          </div>

          {content.diff_details && (
            <div className="space-y-1">
              <span className="text-xs font-medium text-gray-600">
                Changes:
              </span>
              <pre className="overflow-x-auto whitespace-pre-wrap rounded bg-gray-100 p-2 text-xs">
                {content.diff_details}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
