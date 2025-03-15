import { useNavigate, useLocation } from "react-router-dom";
import type { FlowConfig } from "@/lib/types/flow";
import { toast } from "sonner";
import { useCreateFlow } from "@/lib/hooks/useFlows";
import { createDefaultFlow } from "@/lib/utils/resourceDefaults";
import { FormPageLayout } from "@/lib/components/layouts/FormPageLayout";
import { FlowForm } from "./components/FlowForm";

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
        <FormPageLayout className="max-w-4xl">
            <FlowForm
                flow={defaultFlow}
                onSubmit={handleSubmit}
                isNew={true}
            />
        </FormPageLayout>
    );
} 