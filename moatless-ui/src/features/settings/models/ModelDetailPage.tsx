import { useParams, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useModel, useUpdateModel, useTestModel, useDeleteModel } from '@/lib/hooks/useModels';
import { Loader2, PlayCircle, Trash2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/lib/components/ui/alert';
import { FormPageLayout } from '@/lib/components/layouts/FormPageLayout';
import { FormContainer } from '@/lib/components/form/settings-form';
import { Button } from '@/lib/components/ui/button';
import { useState } from 'react';
import { modelFormSections, ModelFormSchema } from './types';
import { FormSchema } from '@/lib/components/form/types';
import { TestModelModal } from './components/TestModelModal';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/lib/components/ui/alert-dialog';

export function ModelDetailPage() {
    const { id } = useParams();
    const navigate = useNavigate();
    const { data: model, isLoading, error } = useModel(id!);
    const updateModelMutation = useUpdateModel();
    const testModelMutation = useTestModel();
    const deleteModelMutation = useDeleteModel();
    const [testModalOpen, setTestModalOpen] = useState(false);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

    // Function to handle test model button click
    const handleTestModel = async () => {
        if (!id) return;

        setTestModalOpen(true);
        if (!testModelMutation.data) {
            try {
                await testModelMutation.mutateAsync(id);
            } catch (error) {
                toast.error('Failed to test model');
            }
        }
    };

    // Function to handle model deletion
    const handleDeleteModel = async () => {
        if (!id) return;

        try {
            await deleteModelMutation.mutateAsync(id);
            toast.success('Model deleted successfully');
            navigate('/settings/models');
        } catch (error) {
            toast.error('Failed to delete model');
            console.error('Error deleting model:', error);
        }
    };

    const handleSubmit = async (formData: any) => {
        if (!model) return;

        try {
            // Merge the form data with the model ID to ensure it's included
            const updatedModel = {
                ...formData,
                model_id: model.model_id
            };

            await updateModelMutation.mutateAsync(updatedModel);
            toast.success('Changes saved successfully');
        } catch (error) {
            // Better error handling
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to save changes';

            toast.error(errorMessage);
        }
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
                    <AlertTitle>Error Loading Model</AlertTitle>
                    <AlertDescription>
                        {error instanceof Error ? error.message : 'Failed to load model'}
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    if (!model) {
        return (
            <div className="flex h-full w-full items-center justify-center">
                <Alert className="max-w-md">
                    <AlertTitle>Model Not Found</AlertTitle>
                    <AlertDescription>
                        The requested model could not be found.
                    </AlertDescription>
                </Alert>
            </div>
        );
    }

    // Convert the model form sections to the FormSchema format
    const formSchema: FormSchema = {
        id: model.model_id,
        title: model.model_id,
        sections: modelFormSections,
    };

    // Create action buttons
    const actionButtons = (
        <>
            <Button
                type="button"
                variant="outline"
                onClick={handleTestModel}
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
    );

    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {updateModelMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {updateModelMutation.error instanceof Error
                            ? updateModelMutation.error.message
                            : 'Failed to save changes'}
                    </AlertDescription>
                </Alert>
            )}

            <FormContainer
                schema={formSchema}
                initialValues={model}
                onSave={handleSubmit}
                actionButtons={actionButtons}
                zodSchema={ModelFormSchema}
            />

            {/* Test Model Modal */}
            <TestModelModal
                open={testModalOpen}
                onOpenChange={setTestModalOpen}
                modelId={model.model_id}
                isLoading={testModelMutation.isPending}
                testResult={testModelMutation.data}
            />

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
        </FormPageLayout>
    );
} 