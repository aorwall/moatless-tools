import { useLastUsedStore } from "@/lib/stores/lastUsedStore";

export function useModelSelection() {
  const { lastUsedModel, setLastUsedModel } = useLastUsedStore();

  const handleModelSelect = (modelId: string) => {
    setLastUsedModel(modelId);
    return modelId;
  };

  return {
    selectedModelId: lastUsedModel,
    onModelSelect: handleModelSelect,
  };
} 