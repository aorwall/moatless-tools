import { useQuery } from "@tanstack/react-query";
import { settingsApi } from "@/lib/api/settings";

export const componentKeys = {
  all: ["components"] as const,
  selectors: () => [...componentKeys.all, "selectors"] as const,
  valueFunctions: () => [...componentKeys.all, "value-functions"] as const,
  feedbackGenerators: () =>
    [...componentKeys.all, "feedback-generators"] as const,
  artifactHandlers: () => [...componentKeys.all, "artifact-handlers"] as const,
  memory: () => [...componentKeys.all, "memory"] as const,
  actions: () => [...componentKeys.all, "actions"] as const,
  byType: (type: string) => [...componentKeys.all, type] as const,
};

/**
 * Generic hook to fetch components by type
 * @param componentType The type of component to fetch (e.g., "selectors", "value-functions")
 */
export function useComponents(componentType: string) {
  return useQuery({
    queryKey: componentKeys.byType(componentType),
    queryFn: () => settingsApi.getComponentsByType(componentType),
  });
}

export function useSelectors() {
  return useQuery({
    queryKey: componentKeys.selectors(),
    queryFn: () => settingsApi.getAvailableSelectors(),
  });
}


export function useMemory() {
  return useQuery({
    queryKey: componentKeys.memory(),
    queryFn: () => settingsApi.getAvailableMemory(),
  });
}

export function useValueFunctions() {
  return useQuery({
    queryKey: componentKeys.valueFunctions(),
    queryFn: () => settingsApi.getAvailableValueFunctions(),
  });
}

export function useFeedbackGenerators() {
  return useQuery({
    queryKey: componentKeys.feedbackGenerators(),
    queryFn: () => settingsApi.getAvailableFeedbackGenerators(),
  });
}

export function useArtifactHandlers() {
  return useQuery({
    queryKey: componentKeys.artifactHandlers(),
    queryFn: () => settingsApi.getAvailableArtifactHandlers(),
  });
}

export function useActions() {
  return useQuery({
    queryKey: componentKeys.actions(),
    queryFn: () => settingsApi.getComponentsByType("actions"),
  });
}
