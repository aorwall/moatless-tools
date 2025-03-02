import { useParams } from "react-router-dom";
import { ModelDetail } from "@/pages/settings/models/components/ModelDetail";
import type { ModelConfig } from "@/lib/types/model";
import { toast } from "sonner";
import { useModel, useUpdateModel } from "@/lib/hooks/useModels";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";

export function ModelDetailPage() {
  const { id } = useParams();
  const { data: model, isLoading, error } = useModel(id!);
  const updateModelMutation = useUpdateModel();

  const handleSubmit = async (formData: ModelConfig) => {
    try {
      await updateModelMutation.mutateAsync({
        ...formData,
        id: id!,
      });
      toast.success("Changes saved successfully");
    } catch (error) {
      // Better error handling
      const errorMessage =
        error instanceof Error
          ? error.message
          : (error as any)?.response?.data?.detail || "Failed to save changes";

      toast.error(errorMessage);
      throw error;
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
            {error instanceof Error ? error.message : "Failed to load model"}
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

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        <ModelDetail model={model} onSubmit={handleSubmit} />
      </div>
    </div>
  );
}
