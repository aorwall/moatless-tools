import { useParams } from "react-router-dom";
import { useModel, useUpdateModel } from "@/lib/hooks/useModels";
import { ModelDetail } from "./components/ModelDetail";
import { toast } from "sonner";
import type { ModelConfig } from "@/lib/types/model";

export function ModelsPage() {
  const { id } = useParams();
  const updateModelMutation = useUpdateModel();

  // Don't try to load model details for the base models view
  if (id === "base") {
    return null;
  }

  const { data: selectedModel } = useModel(id ?? "");

  const handleSubmit = async (formData: ModelConfig) => {
    try {
      await updateModelMutation.mutateAsync({
        ...formData,
        id: id!,
      });
      toast.success("Changes saved successfully");
    } catch (error) {
      const errorMessage = error instanceof Error 
        ? error.message 
        : (error as any)?.response?.data?.detail || "Failed to save changes";
      toast.error(errorMessage);
      throw error;
    }
  };

  if (!selectedModel) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          Select a model to view details
        </div>
      </div>
    );
  }

  return <ModelDetail model={selectedModel} onSubmit={handleSubmit} />;
}
