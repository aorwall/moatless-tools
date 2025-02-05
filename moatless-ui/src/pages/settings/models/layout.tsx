import { Outlet, useNavigate, useParams } from "react-router-dom";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useModels } from "@/lib/hooks/useModels";
import type { ModelConfig } from "@/lib/types/model";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";

export function ModelsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: models, isLoading, error } = useModels();

  const filterFields = [
    { name: "model", type: "text" as const },
    {
      name: "response_format",
      type: "select" as const,
      options: ["TOOL_CALL", "REACT"],
    },
  ];

  const getModelDisplay = (model: ModelConfig) => ({
    title: model.model,
    subtitle: `${model.response_format}`,
  });

  const handleModelSelect = (model: ModelConfig) => {
    navigate(`/settings/models/${encodeURIComponent(model.id)}`);
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
          <AlertTitle>Error Loading Models</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "Failed to load models"}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!models?.length) {
    return (
      <div className="flex h-full w-full items-center justify-center">
        <Alert className="max-w-md">
          <AlertTitle>No Models Available</AlertTitle>
          <AlertDescription>
            No models have been configured yet.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const modelList = (
    <DataExplorer
      items={models}
      filterFields={filterFields}
      itemDisplay={getModelDisplay}
      onSelect={handleModelSelect}
      selectedItem={models.find((m) => m.id === id)}
    />
  );

  return <SplitLayout left={modelList} right={<Outlet />} />;
}
