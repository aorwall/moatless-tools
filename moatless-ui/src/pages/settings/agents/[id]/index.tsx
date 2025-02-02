import { useParams } from 'react-router-dom';
import { useAgent, useUpdateAgent } from '@/lib/hooks/useAgents';
import { ActionSelector } from './components/ActionSelector';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/lib/components/ui/tabs';
import { Button } from '@/lib/components/ui/button';
import { Textarea } from '@/lib/components/ui/textarea';
import { toast } from 'sonner';
import { useState, useEffect } from 'react';
import { SelectedActions } from './components/SelectedActions';
import { ActionConfig } from '@/lib/types/agent';
import { useActionStore } from '@/lib/stores/actionStore';
import { createActionConfigFromSchema } from './utils/actionUtils';

export function AgentDetailPage() {
  const { id } = useParams();
  const { data: agent, isLoading } = useAgent(id ?? '');
  const updateAgent = useUpdateAgent();
  const [selectedActions, setSelectedActions] = useState<ActionConfig[]>([]);
  const { getActionByClassName } = useActionStore();
  const [systemPrompt, setSystemPrompt] = useState('');

  useEffect(() => {
    if (agent) {
      setSelectedActions(agent.actions || []);
      setSystemPrompt(agent.system_prompt || '');
    }
  }, [agent]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    
    if (!agent) return;

    try {
      await updateAgent.mutateAsync({
        ...agent,
        system_prompt: systemPrompt,
        actions: selectedActions,
      });
      toast.success('Agent updated successfully');
    } catch (error) {
      console.error('Error updating agent:', error);
      toast.error('Failed to update agent');
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

  if (isLoading) {
    return <div className="flex h-full items-center justify-center">Loading agent...</div>;
  }

  if (!agent) {
    return <div className="text-destructive">Agent not found</div>;
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Agent Configuration: {agent.id}</h1>
            <p className="text-sm text-muted-foreground">Model: {agent.model_id}</p>
          </div>
          <Button 
            type="submit"
            form="agent-form"
            disabled={updateAgent.isPending}
          >
            {updateAgent.isPending ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <form id="agent-form" onSubmit={handleSubmit} className="h-full flex flex-col">
          <Tabs defaultValue="actions" className="flex-1 flex flex-col min-h-0">
            <div className="flex-none border-b bg-muted">
              <TabsList>
                <TabsTrigger value="actions">Actions</TabsTrigger>
                <TabsTrigger value="prompt">System Prompt</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="actions" className="flex-1 p-6 min-h-0">
              <div className="flex gap-6 h-full">
                <div className="flex-1 min-h-0">
                  <SelectedActions
                    actions={selectedActions}
                    onRemove={handleActionRemove}
                    onPropertyChange={handlePropertyChange}
                  />
                </div>
                <div className="flex-1 min-h-0">
                  <ActionSelector
                    selectedActions={selectedActions.map(a => a.action_class)}
                    onSelect={handleActionAdd}
                  />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="prompt" className="flex-1 p-6 min-h-0">
              <div className="flex flex-col h-full">
                <div className="flex-none space-y-2 mb-2">
                  <label 
                    htmlFor="system_prompt" 
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    System Prompt
                  </label>
                  <p className="text-sm text-muted-foreground">
                    The complete system prompt that defines the agent's behavior, role, and guidelines.
                  </p>
                </div>
                <Textarea
                  id="system_prompt"
                  name="system_prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="Enter the system prompt for the agent..."
                  className="flex-1 font-mono text-sm leading-relaxed resize-none min-h-0"
                />
              </div>
            </TabsContent>
          </Tabs>
        </form>
      </div>
    </div>
  );
} 