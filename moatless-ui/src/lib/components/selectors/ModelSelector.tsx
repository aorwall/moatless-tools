import { useModels } from "@/lib/hooks/useModels";
import { GenericSelector, OptionType } from "@/lib/components/GenericSelector";
import { useLastUsedStore } from "@/lib/stores/lastUsedStore";
import { Skeleton } from "@/lib/components/ui/skeleton";
import { Link } from "react-router-dom";
import { Button } from "@/lib/components/ui/button";
import { PlusCircle } from "lucide-react";

interface ModelSelectorProps {
  selectedModelId: string;
  onModelSelect: (id: string) => void;
}

export function ModelSelector({
  selectedModelId,
  onModelSelect,
}: ModelSelectorProps) {
  const { data: modelResponse, isLoading } = useModels();
  const { setLastUsedModel } = useLastUsedStore();
  const models = modelResponse?.models || [];

  const handleSelect = (id: string) => {
    setLastUsedModel(id);
    onModelSelect(id);
  };

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (!models?.length) {
    return (
      <div className="flex items-center gap-2">
        <div className="text-sm text-muted-foreground flex-1">
          No models available
        </div>
        <Button asChild variant="outline" size="sm">
          <Link to="/settings/models" className="flex items-center gap-2">
            <PlusCircle className="h-4 w-4" />
            Add Model
          </Link>
        </Button>
      </div>
    );
  }

  const options: OptionType[] = models.map((model) => ({
    id: model.model_id,
    label: model.model_id,
  }));

  const renderInfo = (selected: OptionType | undefined) => {
    if (!selected) return null;
    const model = models.find((m) => m.model_id === selected.id);
    if (!model) return null;
    return (
      <>
        <p>
          <span className="font-medium">Base Model:</span> {model.model}
        </p>
        <p>
          <span className="font-medium">Temperature:</span> {model.temperature}
        </p>
      </>
    );
  };

  return (
    <GenericSelector
      value={selectedModelId}
      onValueChange={handleSelect}
      placeholder="Select a model"
      options={options}
      renderAdditionalInfo={renderInfo}
    />
  );
}
