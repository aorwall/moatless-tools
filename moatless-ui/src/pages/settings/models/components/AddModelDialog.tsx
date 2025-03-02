import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/lib/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/lib/components/ui/form";
import { Input } from "@/lib/components/ui/input";
import { Button } from "@/lib/components/ui/button";
import { useAddFromBase } from "@/lib/hooks/useModels";
import type { ModelConfig } from "@/lib/types/model";
import { toast } from "sonner";

const addModelSchema = z.object({
  new_model_id: z.string().min(1, "Model ID is required"),
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
      new_model_id: baseModel.id,
    },
  });

  // Reset form when base model changes
  useEffect(() => {
    form.reset({
      new_model_id: baseModel.id,
    });
  }, [form, baseModel.id]);

  const onSubmit = async (data: AddModelFormData) => {
    try {
      setIsSubmitting(true);
      await addFromBase.mutateAsync({
        base_model_id: baseModel.id,
        new_model_id: data.new_model_id,
      });
      toast.success("Model added successfully");
      onOpenChange(false);
      form.reset();
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : (error as any)?.response?.data?.detail || "Failed to add model";
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

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="new_model_id"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Model ID</FormLabel>
                  <FormControl>
                    <Input
                      {...field}
                      placeholder="Enter a unique identifier for your model"
                      autoFocus
                      onFocus={(e) => e.target.select()}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

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
                {isSubmitting ? "Adding..." : "Add Model"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
