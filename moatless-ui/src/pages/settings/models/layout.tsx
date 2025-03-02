import { Outlet, useNavigate, useParams } from "react-router-dom";
import { DataExplorer } from "@/lib/components/DataExplorer";
import { useModels } from "@/lib/hooks/useModels";
import type { ModelConfig } from "@/lib/types/model";
import { Loader2, Plus, Trash2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { SplitLayout } from "@/lib/components/layouts/SplitLayout";
import { Button } from "@/lib/components/ui/button";
import { BaseModelsList } from "./components/BaseModelsList";
import { CreateModelForm } from "./components/CreateModelForm";

export function ModelsLayout() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: userModels, isLoading, error } = useModels();

  const filterFields = [
    { name: "model", type: "text" as const },
    {
      name: "response_format",
      type: "select" as const,
      options: ["tool_call", "react"],
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

  const modelList = (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="font-semibold">Models</h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate("/settings/models/base")}
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Model
        </Button>
      </div>

      {userModels?.models && userModels.models.length > 0 ? (
        <DataExplorer
          items={userModels.models}
          filterFields={filterFields}
          itemDisplay={getModelDisplay}
          onSelect={handleModelSelect}
          selectedItem={userModels.models.find((m) => m.id === id)}
        />
      ) : (
        <div className="flex flex-col items-center justify-center h-full p-4 text-center">
          <p className="text-sm text-gray-500 mb-4">
            No custom models configured
          </p>
          <Button
            variant="outline"
            onClick={() => navigate("/settings/models/base")}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add Model
          </Button>
        </div>
      )}
    </div>
  );

  const getRightContent = () => {
    return <Outlet />;
  };

  return <SplitLayout left={modelList} right={getRightContent()} />;
}
