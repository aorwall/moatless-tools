import {
  MessageSquare,
  Bot,
  Lightbulb,
  Terminal,
  Eye,
  Folder,
  AlertTriangle,
  Cpu,
  LucideIcon,
  ChevronRight,
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
  WorkspaceTrajectoryItem,
  WorkspaceTimelineContent,
} from "@/lib/components/trajectory/items/WorkspaceTrajectoryItem";
import {
  ErrorTrajectoryItem,
  ErrorTimelineContent,
} from "@/lib/components/trajectory/items/ErrorTrajectoryItem";
import { ExpandableItem } from "@/lib/components/trajectory/ExpandableItem";
import { useTrajectoryStore } from "@/pages/trajectory/stores/trajectoryStore";
import { Icons } from "@/lib/utils/icon-mappings";
import {
  ArtifactTrajectoryItem,
  ArtifactTimelineContent,
} from "@/lib/components/trajectory/items/ArtifactTrajectoryItem";
import { TIMELINE_CONFIG } from './Timeline';  // You'll need to export it from Timeline
import { cn } from "@/lib/utils";

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
    setSelectedItem({
      instanceId,
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
      case "workspace":
        return !!(
          content.updatedFiles?.length ||
          content.testResults?.length ||
          content.files?.length
        );
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
      case "workspace":
        return (
          <WorkspaceTrajectoryItem
            content={content as WorkspaceTimelineContent}
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
      default:
        return null;
    }
  })();

  const Icon = getIcon(type);

  return (
    <div 
      onClick={handleSelect} 
      className={cn(
        "cursor-pointer relative group/item",
        "hover:bg-white hover:shadow-sm hover:border-gray-200",
        "rounded-lg transition-all duration-200",
        "p-3 -mx-3", // Negative margin to allow hover effect to extend
        {
          "mb-4": !isLast // Space between items
        }
      )}
    >
      {/* Add subtle hover effect indicator */}
      <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover/item:opacity-100 transition-opacity">
        <ChevronRight className="h-4 w-4 text-gray-400" />
      </div>
      
      {/* Vertical line extending down if not last or if parent has more nodes */}
      {(!isLast || hasNextSibling) && (
        <div 
          className="absolute w-px bg-gray-200"
          style={{
            left: TIMELINE_CONFIG.lines.item.offset,
            top: TIMELINE_CONFIG.lines.item.verticalOffset,
            bottom: `-${TIMELINE_CONFIG.nodeSpacing.default}`
          }}
        />
      )}
      
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
  );
};
