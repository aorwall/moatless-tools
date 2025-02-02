import { Outlet, useNavigate, useParams } from 'react-router-dom';
import { DataExplorer } from '@/lib/components/DataExplorer';
import { useAgents } from '@/lib/hooks/useAgents';
import type { AgentConfig } from '@/lib/types/agent';
import { Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { SplitLayout } from '@/lib/components/layouts/SplitLayout';

export function AgentsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: agents, isLoading, error } = useAgents();

  const filterFields = [
    { name: 'model_id', type: 'text' as const },
    { 
      name: 'response_format', 
      type: 'select' as const, 
      options: ['TOOL_CALL', 'REACT']
    }
  ];

  const getAgentDisplay = (agent: AgentConfig) => ({
    title: agent.id,
    subtitle: `Model: ${agent.model_id}`
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
            {error instanceof Error ? error.message : 'Failed to load agents'}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!agents?.length) {
    return (
      <Outlet />
    );
  }

  const agentList = (
    <DataExplorer
      items={agents}
      filterFields={filterFields}
      itemDisplay={getAgentDisplay}
      onSelect={handleAgentSelect}
      selectedItem={agents.find(a => a.id === id)}
    />
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