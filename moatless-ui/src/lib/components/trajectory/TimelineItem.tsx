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
import './timeline.css';  // Import CSS directly
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

  const renderedContent = (() => {
    switch (type) {
      case "user_message":
      case "assistant_message":
      case "thought":
        return (
          <MessageTrajectoryItem
            content={content as MessageTimelineContent}
            type={type}
          />
        );
      case "action":
        // Determine the correct action name from the content structure
        
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
          <ErrorTrajectoryItem
            content={content as ErrorTimelineContent}
          />
        );
      case "artifact":
        return (
          <ArtifactTrajectoryItem
            content={content as ArtifactTimelineContent}
          />
        );
      case "reward":
        return (
          <RewardTrajectoryItem
            content={content as RewardTimelineContent}
          />
        );
      default:
        return null;
    }
  })();

  const Icon = getIcon(type);

  return (
    <div onClick={handleSelect} className="cursor-pointer relative group">    
      <div className={cn(
        "relative py-2 transition-colors duration-150",
        "hover:bg-gray-50/50 rounded-md -mx-2 px-2",
        "group/item"
      )}>
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
