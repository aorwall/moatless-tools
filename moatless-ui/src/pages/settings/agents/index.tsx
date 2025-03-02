import { useParams } from "react-router-dom";
import { useAgent, useUpdateAgent } from "@/lib/hooks/useAgents";
import { toast } from "sonner";
import type { AgentConfig } from "@/lib/types/agent";
import { AgentDetail } from "./components/AgentDetail";

export function AgentsPage() {
  const { id } = useParams();
  const updateAgentMutation = useUpdateAgent();

  // Don't try to load agent details for the new agent view
  if (id === "new") {
    return null;
  }

  const { data: selectedAgent } = useAgent(id ?? "");

  const handleSubmit = async (formData: AgentConfig) => {
    try {
      await updateAgentMutation.mutateAsync({
        ...formData,
        id: id!,
      });
      toast.success("Changes saved successfully");
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : (error as any)?.response?.data?.detail || "Failed to save changes";
      toast.error(errorMessage);
      throw error;
    }
  };

  if (!selectedAgent) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          Select an agent to view details
        </div>
      </div>
    );
  }

  return <AgentDetail agent={selectedAgent} onSubmit={handleSubmit} />;
}
