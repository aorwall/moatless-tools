import { MessageDetails } from "./MessageDetails.tsx";
import { ActionDetails } from "./ActionDetails.tsx";
import { ObservationDetails } from "./ObservationDetails.tsx";
import { CompletionDetails } from "./CompletionDetails.tsx";
import { WorkspaceFilesDetails } from "./WorkspaceFilesDetails.tsx";
import { WorkspaceContextDetails } from "./WorkspaceContextDetails.tsx";
import { WorkspaceTestsDetails } from "./WorkspaceTestsDetails.tsx";
import { ErrorDetails } from "./ErrorDetails.tsx";
import { ArtifactDetails } from "./ArtifactDetails.tsx";
import { RewardDetails } from "./RewardDetails.tsx";

interface DetailsProps {
  type: string;
  content: any;
  nodeId?: number;
  trajectoryId?: string;
  trajectory?: any;
}

export const Details = ({ type, content, nodeId, trajectoryId, trajectory }: DetailsProps) => {
  switch (type) {
    case "user_message":
    case "assistant_message":
    case "thought":
      return <MessageDetails content={content} type={type} />;
    case "action":
      return <ActionDetails 
        content={content} 
        nodeId={nodeId || 0} 
        trajectory={trajectory || {}} 
      />;
    case "observation":
      return <ObservationDetails content={content} />;
    case "completion":
      return <CompletionDetails content={content} />;
    case "workspace_files":
      return <WorkspaceFilesDetails content={content} />;
    case "workspace_context":
      return <WorkspaceContextDetails content={content} />;
    case "workspace_tests":
      return <WorkspaceTestsDetails content={content} />;
    case "error":
      return <ErrorDetails content={content} />;
    case "artifact":
      return <ArtifactDetails content={content} trajectoryId={trajectoryId || ""} />;
    case "reward":
      return <RewardDetails content={content} />;
    default:
      return <div>Unknown item type: {type}</div>;
  }
}; 