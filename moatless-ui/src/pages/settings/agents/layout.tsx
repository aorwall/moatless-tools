import { Outlet, useNavigate, useParams } from "react-router-dom";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useAgents } from "@/lib/hooks/useAgents";
import type { AgentConfig } from "@/lib/types/agent";
import { Loader2, Plus } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { Button } from "@/lib/components/ui/button";

export function AgentsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: agents = [], isLoading, error } = useAgents();


  const getAgentDisplay = (agent: AgentConfig) => ({
    title: agent.id,
    subtitle: ``,
  });

  const handleAgentSelect = (agent: AgentConfig) => {
    navigate(`/settings/agents/${encodeURIComponent(agent.id)}`);
  };

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full w-full items-center justify-center p-4">
        <Alert variant="destructive" className="max-w-md">
          <AlertTitle>Error Loading Agents</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load agents"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const agentList = (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="font-semibold">Agents</h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate("/settings/agents/new")}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Agent
        </Button>
      </div>

      {agents.length > 0 ? (
        <DataExplorer
          items={agents}
          itemDisplay={getAgentDisplay}
          onSelect={handleAgentSelect}
          selectedItem={agents.find((a) => a.id === id)}
        />
      ) : (
        <div className="flex flex-col items-center justify-center h-full p-4 text-center">
          <p className="text-sm text-gray-500 mb-4">No agents configured</p>
          <Button
            variant="outline"
            onClick={() => navigate("/settings/agents/new")}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Agent
          </Button>
        </div>
      )}
    </div>
  );

  return (
    <SplitLayout
      left={agentList}
      right={
        <div className="h-full min-h-0 overflow-hidden">
          <Outlet />
        </div>
      }
    />
  );
}
