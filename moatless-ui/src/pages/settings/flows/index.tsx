import { useParams } from "react-router-dom";
import { useFlow, useUpdateFlow } from "@/lib/hooks/useFlows";
import { toast } from "sonner";
import type { FlowConfig } from "@/lib/types/flow";
import { FlowDetail } from "./components/FlowDetail";

export function FlowsPage() {
  const { id } = useParams();
  const updateFlowMutation = useUpdateFlow();

  // Don't try to load flow details for the new flow view
  if (id === "new") {
    return null;
  }

  const { data: selectedFlow } = useFlow(id ?? "");

  const handleSubmit = async (formData: FlowConfig) => {
    try {
      await updateFlowMutation.mutateAsync({
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

  if (!selectedFlow) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          Select a flow to view details
        </div>
      </div>
    );
  }

  return <FlowDetail flow={selectedFlow} onSubmit={handleSubmit} />;
} 