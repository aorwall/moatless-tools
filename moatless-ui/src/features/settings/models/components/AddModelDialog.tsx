import { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/lib/components/ui/dialog';
import { FormField } from '@/lib/components/form/form-field';
import { Input } from '@/lib/components/ui/input';
import { Button } from '@/lib/components/ui/button';
import { useAddFromBase } from '@/lib/hooks/useModels';
import type { ModelConfig } from '../types';
import { toast } from 'sonner';

const addModelSchema = z.object({
    new_model_id: z.string().min(1, 'Model ID is required'),
});

type AddModelFormData = z.infer<typeof addModelSchema>;

interface AddModelDialogProps {
    baseModel: ModelConfig;
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

export function AddModelDialog({
    baseModel,
    open,
    onOpenChange,
}: AddModelDialogProps) {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const addFromBase = useAddFromBase();

    const form = useForm<AddModelFormData>({
        resolver: zodResolver(addModelSchema),
        defaultValues: {
            new_model_id: baseModel.model_id,
        },
    });

    // Reset form when base model changes
    useEffect(() => {
        form.reset({
            new_model_id: baseModel.model_id,
        });
    }, [form, baseModel.model_id]);

    const onSubmit = async (data: AddModelFormData) => {
        try {
            setIsSubmitting(true);
            await addFromBase.mutateAsync({
                base_model_id: baseModel.model_id,
                new_model_id: data.new_model_id,
            });
            toast.success('Model added successfully');
            onOpenChange(false);
            form.reset();
        } catch (error) {
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || 'Failed to add model';
            toast.error(errorMessage);
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Add Model from Base</DialogTitle>
                    <DialogDescription>
                        Create a new model based on {baseModel.model}
                    </DialogDescription>
                </DialogHeader>

                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                    <FormField
                        label="Model ID"
                        htmlFor="new_model_id"
                        tooltip="Enter a unique identifier for your model"
                    >
                        <Input
                            id="new_model_id"
                            {...form.register('new_model_id')}
                            placeholder="Enter a unique identifier for your model"
                            autoFocus
                            onFocus={(e) => e.target.select()}
                        />
                        {form.formState.errors.new_model_id && (
                            <p className="text-sm text-red-500 mt-1">
                                {form.formState.errors.new_model_id.message}
                            </p>
                        )}
                    </FormField>

                    <DialogFooter>
                        <Button
                            type="button"
                            variant="outline"
                            onClick={() => onOpenChange(false)}
                            disabled={isSubmitting}
                        >
                            Cancel
                        </Button>
                        <Button type="submit" disabled={isSubmitting}>
                            {isSubmitting ? 'Adding...' : 'Add Model'}
                        </Button>
                    </DialogFooter>
                </form>
            </DialogContent>
        </Dialog>
    );
} 