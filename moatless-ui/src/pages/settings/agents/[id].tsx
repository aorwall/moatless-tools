import { useParams } from 'react-router-dom';
import { useAgent, useUpdateAgent } from '@/lib/hooks/useAgents';
import { ActionSelector } from './components/ActionSelector';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/lib/components/ui/tabs';
import { Button } from '@/lib/components/ui/button';
import { Textarea } from '@/lib/components/ui/textarea';
import { toast } from 'sonner';

export function AgentDetailPage() {
  const { id } = useParams();
  const { data: agent, isLoading } = useAgent(id ?? '');
  const updateAgent = useUpdateAgent();

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    
    if (!agent) return;

    try {
      await updateAgent.mutateAsync({
        ...agent,
        system_prompt: formData.get('system_prompt') as string,
      });
      toast.success('Agent updated successfully');
    } catch (error) {
      console.error('Error updating agent:', error);
      toast.error('Failed to update agent');
    }
  };

  const handleActionsChange = (actions: string[]) => {
    if (!agent) return;
    
    updateAgent.mutate({
      ...agent,
      actions: actions,
    });
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
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <form onSubmit={handleSubmit} className="space-y-6 p-6">
          <Tabs defaultValue="actions">
            <div className="border-b bg-muted">
              <TabsList>
                <TabsTrigger value="actions">Actions</TabsTrigger>
                <TabsTrigger value="prompt">System Prompt</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="actions" className="pt-6">
              <div className="max-h-[70vh] overflow-y-auto">
                <ActionSelector 
                  selectedActions={agent.actions ?? []}
                  onChange={handleActionsChange}
                />
              </div>
            </TabsContent>

            <TabsContent value="prompt" className="space-y-6 pt-6">
              <div className="space-y-2">
                <label 
                  htmlFor="system_prompt" 
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  System Prompt
                </label>
                <Textarea
                  id="system_prompt"
                  name="system_prompt"
                  defaultValue={agent.system_prompt}
                  placeholder="Enter the system prompt for the agent..."
                  rows={20}
                  className="font-mono text-sm leading-relaxed"
                />
                <p className="text-sm text-muted-foreground">
                  The complete system prompt that defines the agent's behavior, role, and guidelines.
                </p>
              </div>

              <div className="flex justify-end">
                <Button 
                  type="submit" 
                  disabled={updateAgent.isPending}
                >
                  {updateAgent.isPending ? 'Saving...' : 'Save Changes'}
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </form>
      </div>
    </div>
  );
} 