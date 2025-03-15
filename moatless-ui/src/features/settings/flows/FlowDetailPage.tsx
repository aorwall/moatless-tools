import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useFlow, useUpdateFlow } from '@/lib/hooks/useFlows';
import { Loader2, Copy } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { FormPageLayout } from '@/lib/components/layouts/FormPageLayout';
import { FlowForm } from './components/FlowForm';

export function FlowDetailPage() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { data: flow, isLoading, error } = useFlow(id!);
    const updateFlowMutation = useUpdateFlow();

    const handleSubmit = async (formData: any) => {
        if (!flow) return;

        console.log('FlowDetailPage handleSubmit called with:', formData);

        try {
            // Merge the form data with the flow ID to ensure it's included
            const updatedFlow = {
                ...formData,
                id: flow.id
            };

            console.log('Submitting updated flow to API:', updatedFlow);
            await updateFlowMutation.mutateAsync(updatedFlow);
            console.log('Flow update successful');
            toast.success('Changes saved successfully');
        } catch (error) {
            console.error('Error updating flow:', error);
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to save changes';

            toast.error(errorMessage);
        }
    };

    const handleDuplicate = () => {
        if (!flow) return;

        // Create a duplicate flow with a new ID
        const duplicatedFlow = {
            ...flow,
            id: `${flow.id}-copy`,
            description: flow.description
                ? `${flow.description} (Copy)`
                : 'Copy of flow',
        };

        // Navigate to the new flow page with the duplicated flow data
        navigate('/settings/flows/new', {
            state: { duplicatedFlow },
        });
    };

    // Show loading state while initial data is being fetched
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
                    <AlertTitle>Error Loading Flow</AlertTitle>
                    <AlertDescription>
                        {error instanceof Error ? error.message : 'Failed to load flow'}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (!flow) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Alert className="max-w-md">
                    <AlertTitle>Flow Not Found</AlertTitle>
                    <AlertDescription>
                        The requested flow could not be found.
                    </AlertDescription>
                </Alert>
            </div>
        );
    }


    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {updateFlowMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {updateFlowMutation.error instanceof Error
                            ? updateFlowMutation.error.message
                            : 'Failed to save changes'}
                    </AlertDescription>
                </Alert>
            )}
            <FlowForm
                flow={flow}
                onSubmit={handleSubmit}
                onDuplicate={handleDuplicate}
                isNew={false}

            />
        </FormPageLayout>
    );
} 