import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUpdateAgent, useCreateAgent, useDeleteAgent } from '@/lib/hooks/useAgents';
import { Button } from '@/lib/components/ui/button';
import { Textarea } from '@/lib/components/ui/textarea';
import { toast } from 'sonner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/lib/components/ui/tabs';
import { ActionSelector } from '../[id]/components/ActionSelector';
import { SelectedActions } from '../[id]/components/SelectedActions';
import { ActionConfig, AgentConfig } from '@/lib/types/agent';
import { useActionStore } from '@/lib/stores/actionStore';
import { createActionConfigFromSchema } from '../[id]/utils/actionUtils';
import { Input } from '@/lib/components/ui/input';
import { Label } from '@/lib/components/ui/label';

interface AgentFormProps {
  agent?: AgentConfig;
  isNew?: boolean;
}

export function AgentForm({ agent, isNew = false }: AgentFormProps) {
  const navigate = useNavigate();
  const updateAgent = useUpdateAgent();
  const createAgent = useCreateAgent();
  const deleteAgent = useDeleteAgent();
  const [selectedActions, setSelectedActions] = useState<ActionConfig[]>([]);
  const { getActionByClassName } = useActionStore();
  const [systemPrompt, setSystemPrompt] = useState('');
  const [agentId, setAgentId] = useState('');

  useEffect(() => {
    if (agent) {
      setSelectedActions(agent.actions || []);
      setSystemPrompt(agent.system_prompt || '');
      setAgentId(agent.id);
    }
  }, [agent]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    
    const agentData = {
      id: agentId,
      model_id: agent?.model_id || 'gpt-4',
      response_format: agent?.response_format || 'TOOL_CALL',
      system_prompt: systemPrompt,
      actions: selectedActions,
    };

    try {
      if (isNew) {
        const newAgent = await createAgent.mutateAsync(agentData);
        toast.success('Agent created successfully');
        navigate(`/settings/agents/${encodeURIComponent(newAgent.id)}`);
      } else {
        await updateAgent.mutateAsync(agentData);
        toast.success('Agent updated successfully');
      }
    } catch (error) {
      console.error('Error saving agent:', error);
      toast.error(`Failed to ${isNew ? 'create' : 'update'} agent`);
    }
  };

  const handleDelete = async () => {
    if (!agent || !window.confirm('Are you sure you want to delete this agent?')) return;

    try {
      await deleteAgent.mutateAsync(agent.id);
      toast.success('Agent deleted successfully');
      navigate('/settings/agents');
    } catch (error) {
      console.error('Error deleting agent:', error);
      toast.error('Failed to delete agent');
    }
  };

  const handlePropertyChange = (className: string, property: string, value: any) => {
    const newActions = selectedActions.map(action => {
      if (action.action_class === className) {
        return {
          ...action,
          properties: {
            ...action.properties,
            [property]: value
          }
        };
      }
      return action;
    });
    setSelectedActions(newActions);
  };

  const handleActionRemove = (actionClassName: string) => {
    const newActions = selectedActions.filter(a => a.action_class !== actionClassName);
    setSelectedActions(newActions);
  };

  const handleActionAdd = (actionClassName: string) => {
    const actionSchema = getActionByClassName(actionClassName);
    if (!actionSchema) return;

    const newAction = createActionConfigFromSchema(actionSchema);
    const newActions = [...selectedActions, newAction];
    setSelectedActions(newActions);
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              {isNew ? 'Create New Agent' : `Agent: ${agent?.id}`}
            </h1>
          </div>
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => navigate('/settings/agents')}
            >
              Cancel
            </Button>
            {!isNew && (
              <Button
                type="button"
                variant="destructive"
                onClick={handleDelete}
                disabled={deleteAgent.isPending}
              >
                {deleteAgent.isPending ? 'Deleting...' : 'Delete Agent'}
              </Button>
            )}
            <Button 
              type="submit"
              form="agent-form"
              disabled={updateAgent.isPending || createAgent.isPending || (isNew && !agentId.trim())}
            >
              {(updateAgent.isPending || createAgent.isPending) ? 'Saving...' : 'Save Changes'}
            </Button>
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <form id="agent-form" onSubmit={handleSubmit} className="h-full flex flex-col">
          {isNew && (
            <div className="flex-none p-6 border-b">
              <div className="space-y-2 max-w-md">
                <Label htmlFor="agent-id">Agent ID</Label>
                <Input
                  id="agent-id"
                  required
                  value={agentId}
                  onChange={(e) => setAgentId(e.target.value)}
                  placeholder="Enter a unique identifier for the agent"
                />
              </div>
            </div>
          )}

          <Tabs defaultValue="actions" className="flex-1 flex flex-col min-h-0">
            <div className="flex-none border-b bg-muted">
              <TabsList>
                <TabsTrigger value="actions">Actions</TabsTrigger>
                <TabsTrigger value="prompt">System Prompt</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="actions" className="flex-1 p-6 min-h-0">
              <div className="flex gap-6 h-full">
                <div className="flex-1 min-h-0 h-full">
                  <SelectedActions
                    actions={selectedActions}
                    onRemove={handleActionRemove}
                    onPropertyChange={handlePropertyChange}
                  />
                </div>
                <div className="flex-1 min-h-0 h-full">
                  <ActionSelector
                    selectedActions={selectedActions.map(a => a.action_class)}
                    onSelect={handleActionAdd}
                  />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="prompt" className="flex-1 p-6 min-h-0 overflow-hidden">
              <div className="flex flex-col h-full min-h-0">
                <div className="flex-none space-y-2 mb-2">
                  <Label htmlFor="system_prompt">System Prompt</Label>
                  <p className="text-sm text-muted-foreground">
                    The complete system prompt that defines the agent's behavior, role, and guidelines.
                  </p>
                </div>
                <div className="flex-1 min-h-0">
                  <Textarea
                    id="system_prompt"
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    placeholder="Enter the system prompt for the agent..."
                    className="h-full w-full font-mono text-sm leading-relaxed resize-none !min-h-full"
                  />
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </form>
      </div>
    </div>
  );
} 