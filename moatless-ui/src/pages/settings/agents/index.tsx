import { useNavigate } from 'react-router-dom';
import { DataExplorer } from '@/lib/components/DataExplorer';
import { useAgents } from '@/lib/hooks/useAgents';
import type { AgentConfig } from '@/lib/types/agent';
import { Button } from '@/lib/components/ui/button';
import { Plus } from 'lucide-react';

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
    return (
      <div className="flex h-full flex-col items-center justify-center space-y-4 p-4">
        <div className="text-center">
          <h2 className="text-2xl font-bold">No Agents Available</h2>
          <p className="text-muted-foreground">
            Looks like you haven't created any agents yet.
            Click the button below to create your first agent.
          </p>
        </div>
        <Button
          onClick={() => navigate('/settings/agents/new')}
          className="flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          Create First Agent
        </Button>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Agent Configurations</h2>
        <Button 
          onClick={() => navigate('/settings/agents/new')} 
          className="flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Agent
        </Button>
      </div>
      <DataExplorer
        items={agents}
        filterFields={filterFields}
        itemDisplay={getAgentDisplay}
        onSelect={handleAgentSelect}
      />
    </div>
  );
} 