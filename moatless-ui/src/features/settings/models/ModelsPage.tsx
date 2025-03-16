import { useNavigate, useParams } from "react-router-dom";
import { useModels } from "@/lib/hooks/useModels";
import { Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/lib/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/lib/components/ui/card";
import { SettingsHeader } from "@/features/settings/components/SettingsHeader";

export function ModelsPage() {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: userModels, isLoading, error } = useModels();

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

  if (id) {
    // If there's an ID in the URL, the ModelDetailPage will be rendered by the router
    return null;
  }

  return (
    <div className="space-y-6">
      <SettingsHeader
        title="Models"
        description="Manage your AI models and configurations"
        addButtonPath="/settings/models/base"
        addButtonLabel="Add Model"
      />

      {userModels?.models && userModels.models.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {userModels.models.map((model) => (
            <Card
              key={model.model_id}
              className="cursor-pointer hover:bg-accent/50 transition-colors"
              onClick={() => navigate(`/settings/models/${encodeURIComponent(model.model_id)}`)}
            >
              <CardHeader className="pb-2">
                <CardTitle>{model.model_id}</CardTitle>
                <CardDescription>{model.completion_model_class}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Model:</span>
                    <span>{model.model}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Class:</span>
                    <span>{model.completion_model_class}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center p-12 text-center border rounded-lg bg-background">
          <h3 className="text-lg font-medium mb-2">No models configured</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Add a model to get started with custom AI configurations
          </p>
        </div>
      )}
    </div>
  );
}
