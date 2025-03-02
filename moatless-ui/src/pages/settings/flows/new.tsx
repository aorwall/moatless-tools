import { useNavigate, useLocation } from "react-router-dom";
import { FlowDetail } from "@/pages/settings/flows/components/FlowDetail";
import type { FlowConfig } from "@/lib/types/flow";
import { toast } from "sonner";
import { useCreateFlow } from "@/lib/hooks/useFlows";
import {
  createDefaultFlow,
  generateResourceId,
} from "@/lib/utils/resourceDefaults";

export function NewFlowPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const createFlowMutation = useCreateFlow();

  // Check if we have a duplicated flow from the location state
  const duplicatedFlow = location.state?.duplicatedFlow as
    | FlowConfig
    | undefined;

  const defaultFlow: FlowConfig = duplicatedFlow || {
    id: "",
    ...createDefaultFlow(),
  };

  const handleSubmit = async (formData: FlowConfig) => {
    try {
      const newFlow = await createFlowMutation.mutateAsync(formData);
      navigate(`/settings/flows/${encodeURIComponent(newFlow.id)}`, {
        replace: true,
      });
      toast.success("Flow created successfully");
    } catch (error) {
      toast.error("Failed to create flow");
      throw error;
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">New Flow</h1>
            <div className="mt-1 text-sm text-gray-500">
              {duplicatedFlow
                ? "Create a new flow based on an existing configuration"
                : "Create a new flow configuration"}
            </div>
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        <FlowDetail flow={defaultFlow} onSubmit={handleSubmit} isNew />
      </div>
    </div>
  );
}
