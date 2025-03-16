import { Trajectory, Node, ActionStep, FileContextFile } from "@/lib/types/trajectory.ts";
import { NodeCompletionContent } from "./NodeCompletionContent";
import { useTrajectoryStore } from "../../stores/trajectoryStore";
import { useGetNode } from "../../hooks/useGetNode";
import { ActionDetails } from "./ActionDetails";
import { ObservationDetails } from "./ObservationDetails";
import { ActionItem } from "../../components/tree-view/types";
import { RewardDetails } from "./RewardDetails";
import { WorkspaceContextDetails } from "./WorkspaceContextDetails";
import { WorkspaceFilesDetails, WorkspaceFilesTimelineContent } from "./WorkspaceFilesDetails";
import { WorkspaceTestsDetails } from "./WorkspaceTestsDetails";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/lib/components/ui/tabs";
import { ErrorDetails } from "./ErrorDetails";

interface NodeDetailsProps {
  trajectory: Trajectory;
}

export const NodeDetails = ({ trajectory }: NodeDetailsProps) => {
  const selectedTreeItem = useTrajectoryStore((state) =>
    state.getSelectedTreeItem(trajectory.trajectory_id),
  );

  const { data: node, isLoading, error } = useGetNode(
    trajectory.project_id,
    trajectory.trajectory_id,
    selectedTreeItem?.node_id ?? 0
  );

  if (!selectedTreeItem) {
    return <div>No node selected</div>;
  }

  if (selectedTreeItem.type === "completion") {
    console.log("selectedTreeItem", selectedTreeItem);
    return <NodeCompletionContent trajectory={trajectory} nodeId={selectedTreeItem.node_id} actionStep={selectedTreeItem.action_step_id} />;
  }

  if (selectedTreeItem.type === "reward" && node) {
    // Cast node to Node type
    const typedNode = (node as unknown) as Node;

    // Assuming the reward information is in the node
    if (typedNode.reward) {
      return <RewardDetails content={typedNode.reward} />;
    }
  }

  if (selectedTreeItem.type === "error" && node) {
    // Cast node to Node type
    const typedNode = (node as unknown) as Node;

    // Check if there's an error message
    if (typedNode.error) {
      return <ErrorDetails content={{ error: typedNode.error }} />;
    }
  }

  if (selectedTreeItem.type === "action" && node) {
    // Cast node to Node type since we know it's a Node, but need to convert to unknown first
    const typedNode = (node as unknown) as Node;

    if (typedNode.action_steps && typedNode.action_steps.length > 0) {
      // Find the relevant action step using the action index from the selectedTreeItem
      const actionTreeItem = selectedTreeItem as ActionItem;
      const actionStep = typedNode.action_steps[actionTreeItem.action_index];

      if (actionStep) {
        return (
          <div className="space-y-6">
            <ActionDetails
              content={actionStep.action}
              nodeId={selectedTreeItem.node_id}
              trajectory={trajectory}
            />

            {actionStep.observation && (
              <div className="border-t pt-4">
                <h3 className="font-semibold text-sm text-gray-600 mb-3">Observation</h3>
                <ObservationDetails content={actionStep.observation} />
              </div>
            )}
          </div>
        );
      }
    }
  }

  // Default case - show user/assistant message or thought
  const typedNode = (node as unknown) as Node | undefined;

  return (
    <div className="space-y-6">
      {/* User Message, Assistant Message, or Thought */}
      {(typedNode?.user_message || typedNode?.assistant_message || typedNode?.thoughts) && (
        <div className="space-y-2">
          <div className="text-sm text-gray-500">
            {typedNode?.user_message
              ? "User Message"
              : typedNode?.assistant_message
                ? "Assistant Message"
                : "Thought"}
          </div>
          <div className="prose prose-sm max-w-none">
            <pre className="whitespace-pre-wrap rounded-lg bg-gray-50 p-4 text-sm text-gray-700">
              {typedNode?.user_message || typedNode?.assistant_message || typedNode?.thoughts || ""}
            </pre>
          </div>
        </div>
      )}

      {/* File Context Information */}
      {typedNode?.file_context && (
        <div className="mt-6">
          <Tabs defaultValue="files" className="w-full">
            <TabsList className="mb-4">
              {(typedNode.file_context.updatedFiles?.length || 0) > 0 && (
                <TabsTrigger value="files">Updated Files</TabsTrigger>
              )}
              {(typedNode.file_context.files?.length || 0) > 0 && (
                <TabsTrigger value="context">Workspace Context</TabsTrigger>
              )}
              {(typedNode.file_context.testResults?.length || 0) > 0 && (
                <TabsTrigger value="tests">Test Results</TabsTrigger>
              )}
            </TabsList>

            {(typedNode.file_context.updatedFiles?.length || 0) > 0 && (
              <TabsContent value="files">
                <WorkspaceFilesDetails
                  content={{
                    updatedFiles: typedNode.file_context.updatedFiles || [],
                  } as WorkspaceFilesTimelineContent}
                />
              </TabsContent>
            )}

            {(typedNode.file_context.files?.length || 0) > 0 && (
              <TabsContent value="context">
                <WorkspaceContextDetails
                  content={{
                    files: (typedNode.file_context.files || []).map(file => ({
                      file_path: file.file_path,
                      tokens: file.tokens,
                      spans: (file.spans || []).map(span => ({
                        span_id: span.span_id,
                        start_line: span.start_line || 0,
                        end_line: span.end_line || 0,
                        tokens: span.tokens,
                        pinned: span.pinned
                      }))
                    }))
                  }}
                />
              </TabsContent>
            )}

            {(typedNode.file_context.testResults?.length || 0) > 0 && (
              <TabsContent value="tests">
                <WorkspaceTestsDetails
                  content={{
                    test_files: (typedNode.file_context.testResults || []).map(result => ({
                      file_path: result.file_path || 'Unknown file',
                      test_results: result.results || []
                    }))
                  }}
                />
              </TabsContent>
            )}
          </Tabs>
        </div>
      )}
    </div>
  );
};
