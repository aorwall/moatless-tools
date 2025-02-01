import { useNavigate } from 'react-router-dom';
import { DataExplorer } from '@/lib/components/DataExplorer';
import { useAgents } from '@/lib/hooks/useAgents';
import type { AgentConfig } from '@/lib/types/agent';

export function AgentsPage() {
  const navigate = useNavigate();
  const { data: agents, isLoading, error } = useAgents();

  const filterFields = [
    { name: 'model_id', type: 'text' as const },
    { 
      name: 'response_format', 
      type: 'select' as const, 
      options: Array.from(new Set(agents?.map(a => a.response_format) ?? []))
    }
  ];

  const getAgentDisplay = (agent: AgentConfig) => ({
    title: agent.id,
    subtitle: agent.model_id
  });

  const handleAgentSelect = (agent: AgentConfig) => {
    navigate(`/settings/agents/${encodeURIComponent(agent.id)}`);
  };

  if (isLoading) {
    return <div className="flex h-full items-center justify-center">Loading agents...</div>;
  }

  if (error) {
    return <div className="text-destructive">Failed to load agents</div>;
  }

  if (!agents?.length) {
    return <div className="text-muted-foreground">No agents available</div>;
  }

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Agent Configurations</h2>
      <DataExplorer
        items={agents}
        filterFields={filterFields}
        itemDisplay={getAgentDisplay}
        onSelect={handleAgentSelect}
      />
    </div>
  );
} 