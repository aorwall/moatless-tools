import {
  AlertTriangle,
} from "lucide-react";
import {
  ActionTrajectoryItem,
  ActionTimelineContent,
} from "@/lib/components/trajectory/items/ActionTrajectoryItem";
import {
  CompletionTrajectoryItem,
  CompletionTimelineContent,
} from "@/lib/components/trajectory/items/CompletionTrajectoryItem";
import {
  MessageTrajectoryItem,
  MessageTimelineContent,
} from "@/lib/components/trajectory/items/MessageTrajectoryItem";
import {
  ObservationTrajectoryItem,
  ObservationTimelineContent,
} from "@/lib/components/trajectory/items/ObservationTrajectoryItem";

import {
  ErrorTrajectoryItem,
  ErrorTimelineContent,
} from "@/lib/components/trajectory/items/ErrorTrajectoryItem";
import {
  RewardTrajectoryItem,
  RewardTimelineContent,
} from "@/lib/components/trajectory/items/RewardTrajectoryItem";
import { ExpandableItem } from "@/lib/components/trajectory/ExpandableItem";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { Icons } from "@/lib/utils/icon-mappings";
import {
  ArtifactTrajectoryItem,
  ArtifactTimelineContent,
} from "@/lib/components/trajectory/items/ArtifactTrajectoryItem";
import { TIMELINE_CONFIG } from './Timeline';  // You'll need to export it from Timeline
import { cn } from "@/lib/utils";
import {
  WorkspaceFilesTrajectoryItem,
  WorkspaceFilesTimelineContent,
} from "@/lib/components/trajectory/items/WorkspaceFilesTrajectoryItem";
import {
  WorkspaceContextTrajectoryItem,
  WorkspaceContextTimelineContent,
} from "@/lib/components/trajectory/items/WorkspaceContextTrajectoryItem";
import {
  WorkspaceTestsTrajectoryItem,
  WorkspaceTestsTimelineContent,
} from "@/lib/components/trajectory/items/WorkspaceTestsTrajectoryItem";

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

  // Calculate isExpandable based on content type
  const isExpandable = (() => {
    switch (type) {
      case "user_message":
      case "assistant_message":
      case "thought":
        return content.message.length > 200;
      case "action":
        return Object.keys(content.properties || {}).length > 0;
      case "completion":
        return !!(content.input || content.response);
      case "observation":
        return content.message
          ? content.message.length > 300 ||
              content.message.split("\n").length > 5
          : false;
      case "error":
        return content.error.split("\n").length > 1;
      case "reward":
        return !!content.explanation;
      case "workspace_files":
        return !!(content.updatedFiles?.length);
      case "workspace_context":
        return !!(content.files?.length);
      case "workspace_tests":
        return !!(content.test_files?.length);
      default:
        return false;
    }
  })();

  const renderedContent = (() => {
    switch (type) {
      case "user_message":
      case "assistant_message":
      case "thought":
        return (
          <MessageTrajectoryItem
            content={content as MessageTimelineContent}
            type={type}
            expandedState={false}
          />
        );
      case "action":
        return (
          <ActionTrajectoryItem
            content={content as ActionTimelineContent}
            expandedState={false}
          />
        );
      case "observation":
        return (
          <ObservationTrajectoryItem
            content={content as ObservationTimelineContent}
            expandedState={false}
          />
        );
      case "completion":
        return (
          <CompletionTrajectoryItem
            content={content as CompletionTimelineContent}
            expandedState={false}
          />
        );
      case "workspace_files":
        return (
          <WorkspaceFilesTrajectoryItem
            content={content as WorkspaceFilesTimelineContent}
            expandedState={false}
          />
        );
      case "workspace_context":
        return (
          <WorkspaceContextTrajectoryItem
            content={content as WorkspaceContextTimelineContent}
            expandedState={false}
          />
        );
      case "workspace_tests":
        return (
          <WorkspaceTestsTrajectoryItem
            content={content as WorkspaceTestsTimelineContent}
            expandedState={false}
          />
        );
      case "error":
        return (
          <ErrorTrajectoryItem
            content={content as ErrorTimelineContent}
            expandedState={false}
          />
        );
      case "artifact":
        return (
          <ArtifactTrajectoryItem
            content={content as ArtifactTimelineContent}
            expandedState={false}
          />
        );
      case "reward":
        return (
          <RewardTrajectoryItem
            content={content as RewardTimelineContent}
            expandedState={false}
          />
        );
      default:
        return null;
    }
  })();

  const Icon = getIcon(type);

  return (
    <div onClick={handleSelect} className="cursor-pointer relative group">
      {/* Vertical line extending down if not last or if parent has more nodes */}
      {(!isLast || hasNextSibling) && (
        <div 
          className="absolute w-px bg-gray-200/70"
          style={{
            left: TIMELINE_CONFIG.lines.item.offset,
            top: TIMELINE_CONFIG.lines.item.verticalOffset,
            bottom: `-${TIMELINE_CONFIG.nodeSpacing.default}`
          }}
        />
      )}
      
      <div className={cn(
        "relative py-2 transition-colors duration-150",
        "hover:bg-gray-50/50 rounded-md -mx-2 px-2",
        "group/item"
      )}>
        <ExpandableItem
          label={label}
          icon={Icon}
          isExpandable={isExpandable}
          expandedState={false}
          onExpandChange={() => handleSelect()}
        >
          {renderedContent}
        </ExpandableItem>
      </div>
    </div>
  );
};
