import { MessageDetails } from "./MessageDetails";
import { ActionDetails } from "./ActionDetails";
import { ObservationDetails } from "./ObservationDetails";
import { CompletionDetails } from "./CompletionDetails";
import { WorkspaceFilesDetails } from "./WorkspaceFilesDetails";
import { WorkspaceContextDetails } from "./WorkspaceContextDetails";
import { WorkspaceTestsDetails } from "./WorkspaceTestsDetails";
import { ErrorDetails } from "./ErrorDetails";
import { ArtifactDetails } from "./ArtifactDetails";
import { RewardDetails } from "./RewardDetails";

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