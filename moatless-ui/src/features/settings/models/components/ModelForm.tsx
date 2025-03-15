import { useState } from 'react';
import { FormContainer } from '@/lib/components/form/settings-form';
import { Button } from '@/lib/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { Loader2, PlayCircle } from 'lucide-react';
import { type ModelConfig } from '../types';
import { useModelForm } from '../hooks/useModelForm';
import { modelFormSections } from '../types';
import { FormSchema, FormValues } from '@/lib/components/form/types';
import { TestModelModal } from './TestModelModal';

interface ModelFormProps {
    model: ModelConfig;
    onSubmit: (data: ModelConfig) => Promise<void>;
}

export function ModelForm({ model, onSubmit }: ModelFormProps) {
    const [testModalOpen, setTestModalOpen] = useState(false);

    const {
        form,
        isSaving,
        error,
        testModelMutation,
        handleSubmit,
        handleTestModel,
    } = useModelForm({ model, onSubmit });

    // Convert the model form sections to the FormSchema format
    const formSchema: FormSchema = {
        id: model.model_id,
        title: model.model_id,
        sections: modelFormSections,
    };

    // Wrapper function to handle form submission from FormContainer
    const handleFormSubmit = (values: FormValues) => {
        handleSubmit();
    };

    // Function to handle test model button click
    const onTestModelClick = async () => {
        setTestModalOpen(true);
        if (!testModelMutation.data) {
            await handleTestModel();
        }
    };

    // Custom action buttons for the form
    const actionButtons = (
        <Button
            type="button"
            variant="outline"
            onClick={onTestModelClick}
            disabled={testModelMutation.isPending}
            className="gap-2"
        >
            {testModelMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
                <PlayCircle className="h-4 w-4" />
            )}
            {testModelMutation.isPending ? "Testing..." : "Test Model"}
        </Button>
    );

    // If form is validating and not dirty, show loading
    if (form.formState.isValidating && !form.formState.isDirty) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Error message */}
            {error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {/* Form */}
            <FormContainer
                schema={formSchema}
                initialValues={form.getValues()}
                onSave={handleFormSubmit}
                actionButtons={actionButtons}
            />

            {/* Test Model Modal */}
            <TestModelModal
                open={testModalOpen}
                onOpenChange={setTestModalOpen}
                modelId={model.model_id}
                isLoading={testModelMutation.isPending}
                testResult={testModelMutation.data}
            />
        </div>
    );
} 