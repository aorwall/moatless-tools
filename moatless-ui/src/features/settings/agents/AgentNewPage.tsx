import { useNavigate, useLocation } from 'react-router-dom';
import { toast } from 'sonner';
import { useCreateAgent } from '@/lib/hooks/useAgents';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { FormPageLayout } from '@/lib/components/layouts/FormPageLayout';
import { AgentForm } from './components/AgentForm';
import { AgentConfig } from '@/lib/types/agent';

export function AgentNewPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const createAgentMutation = useCreateAgent();

    // Check if we have a duplicated agent from the location state
    const duplicatedAgent = location.state?.duplicatedAgent as AgentConfig | undefined;

    // Create a default agent if none is provided
    const defaultAgent: AgentConfig = duplicatedAgent || {
        agent_id: '',
        system_prompt: '',
        actions: [],
    };

    const handleSubmit = async (formData: AgentConfig) => {

        try {
            const newAgent = await createAgentMutation.mutateAsync(formData);
            toast.success('Agent created successfully');

            navigate(`/settings/agents/${encodeURIComponent(newAgent.agent_id)}`);
        } catch (error) {
            console.error('Error creating agent:', error);
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to create agent';

            toast.error(errorMessage);
        }
    };

    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {createAgentMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {createAgentMutation.error instanceof Error
                            ? createAgentMutation.error.message
                            : 'Failed to create agent'}
                    </AlertDescription>
                </Alert>
            )}
            <AgentForm
                agent={defaultAgent}
                onSubmit={handleSubmit}
                isNew={true}
            />
        </FormPageLayout>
    );
} 