import { useModels } from "@/lib/hooks/useModels";
import { GenericSelector, OptionType } from "@/lib/components/GenericSelector";
import { useLastUsedStore } from "@/lib/stores/lastUsedStore";
import { Skeleton } from "@/lib/components/ui/skeleton";

interface ModelSelectorProps {
  selectedModelId: string;
  onModelSelect: (id: string) => void;
}

export function ModelSelector({
  selectedModelId,
  onModelSelect,
}: ModelSelectorProps) {
  const { data: models, isLoading } = useModels();
  const { setLastUsedModel } = useLastUsedStore();

  const handleSelect = (id: string) => {
    setLastUsedModel(id);
    onModelSelect(id);
  };

  if (isLoading) {
    return <Skeleton className="h-10 w-full" />;
  }

  if (!models?.length) {
    return <div className="text-sm text-muted-foreground">No models available</div>;
  }

  const options: OptionType[] = models.map((model) => ({
    id: model.id,
    label: model.id,
  }));

  const renderInfo = (selected: OptionType | undefined) => {
    if (!selected) return null;
    const model = models.find((m) => m.id === selected.id);
    if (!model) return null;
    return (
      <>
        <p>
          <span className="font-medium">Base Model:</span> {model.model}
        </p>
        <p>
          <span className="font-medium">Response Format:</span>{" "}
          {model.response_format}
        </p>
        <p>
          <span className="font-medium">Temperature:</span> {model.temperature}
        </p>
      </>
    );
  };

  return (
    <GenericSelector
      title="Select Model"
      value={selectedModelId}
      onValueChange={handleSelect}
      placeholder="Select a model"
      options={options}
      renderAdditionalInfo={renderInfo}
    />
  );
}
