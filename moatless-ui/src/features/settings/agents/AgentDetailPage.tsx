import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useAgent, useUpdateAgent } from '@/lib/hooks/useAgents';
import { Loader2, Copy } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { FormPageLayout } from '@/lib/components/layouts/FormPageLayout';
import { AgentForm } from './components/AgentForm';

export function AgentDetailPage() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { data: agent, isLoading, error } = useAgent(id!);
    const updateAgentMutation = useUpdateAgent();

    const handleSubmit = async (formData: any) => {
        if (!agent) return;

        try {
            const updatedAgent = {
                ...formData,
                agent_id: agent.agent_id
            };

            await updateAgentMutation.mutateAsync(updatedAgent);
            toast.success('Changes saved successfully');
        } catch (error) {
            console.error('Error updating agent:', error);
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to save changes';

            toast.error(errorMessage);
        }
    };

    const handleDuplicate = () => {
        if (!agent) return;

        const duplicatedAgent = {
            ...agent,
            agent_id: `${agent.agent_id}-copy`,
        };

        navigate('/settings/agents/new', {
            state: { duplicatedAgent },
        });
    };

    if (isLoading || !id) {
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
                    <AlertTitle>Error Loading Agent</AlertTitle>
                    <AlertDescription>
                        {error instanceof Error ? error.message : 'Failed to load agent'}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (!agent) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Alert className="max-w-md">
                    <AlertTitle>Agent Not Found</AlertTitle>
                    <AlertDescription>
                        The requested agent could not be found.
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {updateAgentMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {updateAgentMutation.error instanceof Error
                            ? updateAgentMutation.error.message
                            : 'Failed to save changes'}
                    </AlertDescription>
                </Alert>
            )}
            <AgentForm
                agent={agent}
                onSubmit={handleSubmit}
                onDuplicate={handleDuplicate}
                isNew={false}
            />
        </FormPageLayout>
    );
} 