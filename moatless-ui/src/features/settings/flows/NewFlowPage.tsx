import { useNavigate, useLocation } from "react-router-dom";
import type { FlowConfig } from "@/lib/types/flow";
import { toast } from "sonner";
import { useCreateFlow } from "@/lib/hooks/useFlows";
import { createDefaultFlow } from "@/lib/utils/resourceDefaults";
import { FormPageLayout } from "@/lib/components/layouts/FormPageLayout";
import { FlowForm } from "./components/FlowForm";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";

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
            toast.success("Flow created successfully");

            navigate(`/settings/flows/${encodeURIComponent(newFlow.id)}`, {
                replace: true,
            });
        } catch (error) {
            console.error("Error creating flow:", error);
            const errorMessage =
                error instanceof Error
                    ? error.message
                    : (error as any)?.response?.data?.detail || "Failed to create flow";

            toast.error(errorMessage);
        }
    };

    return (
        <FormPageLayout className="max-w-4xl">
            {/* Error message */}
            {createFlowMutation.error && (
                <Alert variant="destructive" className="mb-6">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                        {createFlowMutation.error instanceof Error
                            ? createFlowMutation.error.message
                            : "Failed to create flow"}
                    </AlertDescription>
                </Alert>
            )}
            <FlowForm
                flow={defaultFlow}
                onSubmit={handleSubmit}
                isNew={true}
            />
        </FormPageLayout>
    );
} 