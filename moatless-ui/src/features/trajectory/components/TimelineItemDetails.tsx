import { useTrajectoryStore } from "@/features/trajectory/stores/trajectoryStore.ts";
import { ScrollArea } from "@/lib/components/ui/scroll-area.tsx";
import { CompletionDetails } from "@/features/trajectory/components/details/CompletionDetails.tsx";
import { ActionDetails } from "@/features/trajectory/components/details/ActionDetails.tsx";
import { ErrorDetails } from "@/features/trajectory/components/details/ErrorDetails.tsx";
import { MessageDetails } from "@/features/trajectory/components/details/MessageDetails.tsx";
import { ObservationDetails } from "@/features/trajectory/components/details/ObservationDetails.tsx";
import { WorkspaceFilesDetails } from "@/features/trajectory/components/details/WorkspaceFilesDetails.tsx";
import { WorkspaceContextDetails } from "@/features/trajectory/components/details/WorkspaceContextDetails.tsx";
import { WorkspaceTestsDetails } from "@/features/trajectory/components/details/WorkspaceTestsDetails.tsx";
import { ArtifactDetails } from "@/features/trajectory/components/details/ArtifactDetails.tsx";
import { RewardDetails } from "@/features/trajectory/components/details/RewardDetails.tsx";
import { Trajectory } from "@/lib/types/trajectory.ts";

interface TimelineItemDetailsProps {
  trajectoryId: string;
  trajectory: Trajectory;
}

export const TimelineItemDetails = ({
  trajectoryId,
  trajectory,
}: TimelineItemDetailsProps) => {
  const selectedItem = useTrajectoryStore((state) =>
    state.getSelectedItem(trajectoryId),
  );

  if (!selectedItem) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-muted-foreground">
        Select an item to view details
      </div>
    );
  }

  const renderContent = () => {
    switch (selectedItem.type) {
      case "completion":
        return <CompletionDetails content={selectedItem.content} />;
      case "action":
        return (
          <ActionDetails
            content={selectedItem.content}
            nodeId={selectedItem.nodeId}
            trajectory={trajectory}
          />
        );
      case "error":
        return <ErrorDetails content={selectedItem.content} />;
      case "user_message":
      case "assistant_message":
      case "thought":
        return (
          <MessageDetails
            content={selectedItem.content}
            type={selectedItem.type}
          />
        );
      case "observation":
        return <ObservationDetails content={selectedItem.content} />;
      case "workspace_files":
        return <WorkspaceFilesDetails content={selectedItem.content} />;
      case "workspace_context":
        return <WorkspaceContextDetails content={selectedItem.content} />;
      case "workspace_tests":
        return <WorkspaceTestsDetails content={selectedItem.content} />;
      case "artifact":
        return (
          <ArtifactDetails
            content={selectedItem.content}
            trajectoryId={trajectoryId}
          />
        );
      case "reward":
        return <RewardDetails content={selectedItem.content} />;
      default:
        return <div>Unsupported item type</div>;
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b p-4">
        <h2 className="font-semibold">
          {selectedItem.type
            .split("_")
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ")}{" "}
          Details
        </h2>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4">{renderContent()}</div>
      </ScrollArea>
    </div>
  );
};
