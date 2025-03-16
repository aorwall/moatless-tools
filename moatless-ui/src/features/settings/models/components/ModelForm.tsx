import { useState } from 'react';
import { FormContainer } from '@/lib/components/form/settings-form';
import { Button } from '@/lib/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { Loader2, PlayCircle, Trash2 } from 'lucide-react';
import { type ModelConfig } from '../types';
import { useModelForm } from '../hooks/useModelForm';
import { modelFormSections } from '../types';
import { FormSchema, FormValues } from '@/lib/components/form/types';
import { TestModelModal } from './TestModelModal';
import { useDeleteModel } from '@/lib/hooks/useModels';
import { useNavigate } from 'react-router-dom';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/lib/components/ui/alert-dialog";

interface ModelFormProps {
    model: ModelConfig;
    onSubmit: (data: ModelConfig) => Promise<void>;
    isNew?: boolean;
}

export function ModelForm({ model, onSubmit, isNew = false }: ModelFormProps) {
    const [testModalOpen, setTestModalOpen] = useState(false);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const navigate = useNavigate();
    const deleteModelMutation = useDeleteModel();

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
        id: model.model_id || 'new-model',
        title: isNew ? 'Create New Model' : model.model_id,
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

    // Function to handle model deletion
    const handleDeleteModel = async () => {
        try {
            await deleteModelMutation.mutateAsync(model.model_id);
            navigate('/settings/models');
        } catch (error) {
            console.error('Error deleting model:', error);
        }
    };

    // Custom action buttons for the form
    const actionButtons = !isNew ? (
        <>
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
            <Button
                type="button"
                variant="destructive"
                onClick={() => setDeleteDialogOpen(true)}
                disabled={deleteModelMutation.isPending}
                className="gap-2"
            >
                {deleteModelMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                    <Trash2 className="h-4 w-4" />
                )}
                {deleteModelMutation.isPending ? "Deleting..." : "Delete Model"}
            </Button>
        </>
    ) : null;

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

            {/* Test Model Modal - Only show for existing models */}
            {!isNew && (
                <TestModelModal
                    open={testModalOpen}
                    onOpenChange={setTestModalOpen}
                    modelId={model.model_id}
                    isLoading={testModelMutation.isPending}
                    testResult={testModelMutation.data}
                />
            )}

            {/* Delete Confirmation Dialog */}
            <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>Delete Model</AlertDialogTitle>
                        <AlertDialogDescription>
                            Are you sure you want to delete the model "{model.model_id}"? This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={handleDeleteModel}
                            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </div>
    );
} 