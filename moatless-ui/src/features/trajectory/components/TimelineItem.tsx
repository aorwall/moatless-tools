import { AlertTriangle } from "lucide-react";
import {
  ActionTrajectoryItem,
  ActionTimelineContent,
} from "@/features/trajectory/components/items/ActionTrajectoryItem.tsx";
import {
  CompletionTrajectoryItem,
  CompletionTimelineContent,
} from "@/features/trajectory/components/items/CompletionTrajectoryItem.tsx";
import {
  MessageTrajectoryItem,
  MessageTimelineContent,
} from "@/features/trajectory/components/items/MessageTrajectoryItem.tsx";
import {
  ObservationTrajectoryItem,
  ObservationTimelineContent,
} from "@/features/trajectory/components/items/ObservationTrajectoryItem.tsx";

import {
  ErrorTrajectoryItem,
  ErrorTimelineContent,
} from "@/features/trajectory/components/items/ErrorTrajectoryItem.tsx";
import {
  RewardTrajectoryItem,
  RewardTimelineContent,
} from "@/features/trajectory/components/items/RewardTrajectoryItem.tsx";
import { ExpandableItem } from "@/features/trajectory/components/ExpandableItem.tsx";
import { useTrajectoryStore } from "@/features/trajectory/stores/trajectoryStore.ts";
import { Icons } from "@/lib/utils/icon-mappings.ts";
import {
  ArtifactTrajectoryItem,
  ArtifactTimelineContent,
} from "@/features/trajectory/components/items/ArtifactTrajectoryItem.tsx";
import "./timeline.css"; // Import CSS directly
import { cn } from "@/lib/utils.ts";
import {
  WorkspaceFilesTrajectoryItem,
  WorkspaceFilesTimelineContent,
} from "@/features/trajectory/components/items/WorkspaceFilesTrajectoryItem.tsx";
import {
  WorkspaceContextTrajectoryItem,
  WorkspaceContextTimelineContent,
} from "@/features/trajectory/components/items/WorkspaceContextTrajectoryItem.tsx";
import {
  WorkspaceTestsTrajectoryItem,
  WorkspaceTestsTimelineContent,
} from "@/features/trajectory/components/items/WorkspaceTestsTrajectoryItem.tsx";

interface TimelineItemProps {
  type: string;
  content: any;
  nodeId: number;
  instanceId: string;
  itemId: string;
  label: string;
  isLast: boolean;
  hasNextSibling: boolean;
}

// Helper function to safely check if content is valid for rendering
const isValidContent = (content: any): boolean => {
  return content !== null && content !== undefined && typeof content !== 'function';
};

// Helper function to safely stringify objects if needed
const safeStringify = (content: any): string => {
  if (content === null || content === undefined) {
    return '';
  }

  if (typeof content === 'string') {
    return content;
  }

  try {
    return JSON.stringify(content);
  } catch (e) {
    return '[Object]';
  }
};

export const TimelineItem = ({
  type,
  content,
  nodeId,
  instanceId,
  itemId,
  label,
  isLast,
  hasNextSibling,
}: TimelineItemProps) => {
  const { setSelectedItem } = useTrajectoryStore();

  const handleSelect = () => {
    setSelectedItem(instanceId, {
      nodeId,
      itemId,
      type,
      content,
    });
  };

  const getIcon = (type: string) => {
    return Icons[type] || AlertTriangle;
  };

  // Safety check for content
  if (!isValidContent(content)) {
    return (
      <div className="text-red-500 p-2 border border-red-200 rounded">
        Invalid content for item type: {type}
      </div>
    );
  }

  const renderedContent = (() => {
    try {
      switch (type) {
        case "user_message":
        case "assistant_message":
        case "thought":
          return (
            <MessageTrajectoryItem
              content={content as MessageTimelineContent}
              type={type as any}
            />
          );
        case "action":
          return (
            <ActionTrajectoryItem
              name={label}
              content={content as ActionTimelineContent}
            />
          );
        case "observation":
          return (
            <ObservationTrajectoryItem
              content={content as ObservationTimelineContent}
            />
          );
        case "completion":
          return (
            <CompletionTrajectoryItem
              content={content as CompletionTimelineContent}
            />
          );
        case "workspace_context":
          return (
            <WorkspaceContextTrajectoryItem
              content={content as WorkspaceContextTimelineContent}
            />
          );
        case "workspace_files":
          return (
            <WorkspaceFilesTrajectoryItem
              content={content as WorkspaceFilesTimelineContent}
            />
          );
        case "workspace_tests":
          return (
            <WorkspaceTestsTrajectoryItem
              content={content as WorkspaceTestsTimelineContent}
            />
          );
        case "error":
          return (
            <ErrorTrajectoryItem content={content as ErrorTimelineContent} />
          );
        case "artifact":
          return (
            <ArtifactTrajectoryItem
              content={content as ArtifactTimelineContent}
            />
          );
        case "reward":
          return (
            <RewardTrajectoryItem content={content as RewardTimelineContent} />
          );
        default:
          return (
            <div className="p-2 text-gray-500">
              Unknown item type: {type}
            </div>
          );
      }
    } catch (error) {
      console.error("Error rendering timeline item:", error);
      return (
        <div className="p-2 text-red-500 border border-red-200 rounded">
          Error rendering item: {error instanceof Error ? error.message : String(error)}
        </div>
      );
    }
  })();

  const Icon = getIcon(type);

  return (
    <div onClick={handleSelect} className="cursor-pointer relative group">
      <div
        className={cn(
          "relative py-2 transition-colors duration-150",
          "hover:bg-gray-50/50 rounded-md -mx-2 px-2",
          "group/item",
        )}
      >
        <ExpandableItem
          label={label}
          icon={Icon}
          onExpandChange={() => handleSelect()}
        >
          {renderedContent}
        </ExpandableItem>
      </div>
    </div>
  );
};
