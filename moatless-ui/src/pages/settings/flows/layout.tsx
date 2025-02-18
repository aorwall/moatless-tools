import { Outlet, useNavigate, useParams, useLocation } from "react-router-dom";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useFlows, useCreateFlow } from "@/lib/hooks/useFlows";
import type { FlowConfig } from "@/lib/types/flow";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { Button } from "@/lib/components/ui/button";
import { PlusCircle } from "lucide-react";
import { toast } from "sonner";

export function FlowsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const location = useLocation();
  const isNewPage = location.pathname.endsWith('/new');
  const { data: flows, isLoading, error } = useFlows();
  const createFlowMutation = useCreateFlow();

  const filterFields = [
    { name: "id", type: "text" as const },
    { name: "description", type: "text" as const },
  ];

  const getFlowDisplay = (flow: FlowConfig) => ({
    title: flow.id,
    subtitle: flow.description || "No description",
  });

  const handleFlowSelect = (flow: FlowConfig) => {
    navigate(`/settings/flows/${encodeURIComponent(flow.id)}`);
  };

  const handleCreateFlow = () => {
    navigate("/settings/flows/new");
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
          <AlertTitle>Error Loading Flows</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load flows"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const flowList = (
    <div className="flex h-full flex-col">
      <div className="flex-none p-4 border-b">
        <Button onClick={handleCreateFlow}>
          <PlusCircle className="mr-2 h-4 w-4" />
          Create Flow
        </Button>
      </div>
      <div className="flex-1 min-h-0">
        {flows?.length ? (
          <DataExplorer
            items={flows}
            filterFields={filterFields}
            itemDisplay={getFlowDisplay}
            onSelect={handleFlowSelect}
            selectedItem={flows.find((f) => f.id === id)}
          />
        ) : !isNewPage && (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-gray-500">
              No flows configured yet
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return <SplitLayout left={flowList} right={<Outlet />} />;
} 