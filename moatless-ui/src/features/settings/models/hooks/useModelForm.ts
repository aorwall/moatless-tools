import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useState, useEffect, useCallback } from 'react';
import { ModelFormSchema, type ModelFormValues, type ModelConfig } from '../types';
import { useUpdateModel, useTestModel } from '@/lib/hooks/useModels';
import { toast } from 'sonner';

interface UseModelFormProps {
    model: ModelConfig;
    onSubmit?: (data: ModelConfig) => Promise<void>;
}

export function useModelForm({ model, onSubmit }: UseModelFormProps) {
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const testModelMutation = useTestModel();
    const updateModelMutation = useUpdateModel();

    const form = useForm<ModelFormValues>({
        resolver: zodResolver(ModelFormSchema),
        defaultValues: model,
    });

    // Reset form when model changes
    useEffect(() => {
        form.reset(model);
    }, [form, model]);

    const handleSubmit = async (data: ModelFormValues) => {
        try {
            setIsSaving(true);
            setError(null);

            if (onSubmit) {
                await onSubmit(data as ModelConfig);
            } else {
                await updateModelMutation.mutateAsync(data as ModelConfig);
                toast.success('Changes saved successfully');
            }
        } catch (e) {
            const errorMessage =
                e instanceof Error
                    ? e.message
                    : (e as any)?.response?.data?.detail ||
                    'An unexpected error occurred';

            console.error(errorMessage);
            setError(errorMessage);
            throw e;
        } finally {
            setIsSaving(false);
        }
    };

    const handleTestModel = useCallback(async () => {
        try {
            setError(null);
            const result = await testModelMutation.mutateAsync(model.model_id);
            return result;
        } catch (e) {
            const errorMessage =
                e instanceof Error
                    ? e.message
                    : (e as any)?.response?.data?.detail || 'Failed to test model';
            console.error(errorMessage);
            setError(errorMessage);
            toast.error(errorMessage);
            throw e;
        }
    }, [model.model_id, testModelMutation]);

    return {
        form,
        isSaving,
        error,
        testModelMutation,
        handleSubmit: form.handleSubmit(handleSubmit),
        handleTestModel,
    };
} 