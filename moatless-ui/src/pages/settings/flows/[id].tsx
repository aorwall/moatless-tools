import { useParams } from "react-router-dom";
import { FlowDetail } from "@/pages/settings/flows/components/FlowDetail";
import type { FlowConfig } from "@/lib/types/flow";
import { toast } from "sonner";
import { useFlow, useUpdateFlow } from "@/lib/hooks/useFlows";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";

export function FlowDetailPage() {
  const { id } = useParams();
  const { data: flow, isLoading, error } = useFlow(id!);
  const updateFlowMutation = useUpdateFlow();

  const handleSubmit = async (formData: FlowConfig) => {
    try {
      await updateFlowMutation.mutateAsync(formData);
      toast.success("Changes saved successfully");
    } catch (error) {
      toast.error("Failed to save changes");
      throw error;
    }
  };

  if (isLoading) {
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
          <AlertTitle>Error Loading Flow</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load flow"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!flow) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Alert className="max-w-md">
          <AlertTitle>Flow Not Found</AlertTitle>
          <AlertDescription>
            The requested flow could not be found.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{flow.id}</h1>
            <div className="mt-1 text-sm text-gray-500">
              Flow Configuration
            </div>
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-6">
        <FlowDetail flow={flow} onSubmit={handleSubmit} />
      </div>
    </div>
  );
} 