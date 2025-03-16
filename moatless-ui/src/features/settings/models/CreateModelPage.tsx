import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useCreateModel } from '@/lib/hooks/useModels';
import { FormPageLayout } from '@/lib/components/layouts/FormPageLayout';
import { ModelForm } from './components/ModelForm';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { ModelConfig } from '@/lib/types/model';

export function CreateModelPage() {
    const navigate = useNavigate();
    const createModelMutation = useCreateModel();

    // Default model configuration for new models
    const defaultModel: ModelConfig = {
        model_id: '',
        model: '',
        completion_model_class: 'moatless.completion.models.LiteLLMCompletionModel',
        model_base_url: '',
        model_api_key: '',
        temperature: 0.7,
        max_tokens: 4096,
        timeout: 120,
        thoughts_in_action: false,
        disable_thoughts: false,
        merge_same_role_messages: false,
        message_cache: true,
        few_shot_examples: true
    };

    const handleSubmit = async (data: ModelConfig) => {
        try {
            // Map model_id to id for the API
            await createModelMutation.mutateAsync({
                ...data,
                id: data.model_id
            });
            toast.success('Model created successfully');
            navigate(`/settings/models/${encodeURIComponent(data.model_id)}`);
        } catch (error) {
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to create model';
            toast.error(errorMessage);
            throw error; // Re-throw to let the form handle the error state
        }
    };

    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {createModelMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {createModelMutation.error instanceof Error
                            ? createModelMutation.error.message
                            : 'Failed to create model'}
                    </AlertDescription>
                </Alert>
            )}
            <ModelForm
                model={defaultModel}
                onSubmit={handleSubmit}
                isNew={true}
            />
        </FormPageLayout>
    );
} 