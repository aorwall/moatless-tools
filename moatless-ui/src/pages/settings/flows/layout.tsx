import { Outlet, useNavigate, useParams } from "react-router-dom";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useFlows } from "@/lib/hooks/useFlows";
import type { FlowConfig } from "@/lib/types/flow";
import { Loader2, Plus } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { Button } from "@/lib/components/ui/button";

export function FlowsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: flows = [], isLoading, error } = useFlows();

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
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="font-semibold">Flows</h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate("/settings/flows/new")}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Flow
        </Button>
      </div>

      {flows.length > 0 ? (
        <DataExplorer
          items={flows}
          filterFields={filterFields}
          itemDisplay={getFlowDisplay}
          onSelect={handleFlowSelect}
          selectedItem={flows.find((f) => f.id === id)}
        />
      ) : (
        <div className="flex flex-col items-center justify-center h-full p-4 text-center">
          <p className="text-sm text-gray-500 mb-4">No flows configured</p>
          <Button
            variant="outline"
            onClick={() => navigate("/settings/flows/new")}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Flow
          </Button>
        </div>
      )}
    </div>
  );

  return (
    <SplitLayout
      left={flowList}
      right={
        <div className="h-full min-h-0 overflow-hidden">
          <Outlet />
        </div>
      }
    />
  );
}
