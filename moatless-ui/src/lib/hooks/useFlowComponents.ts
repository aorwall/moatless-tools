import { useQuery } from "@tanstack/react-query";
import { settingsApi } from "@/lib/api/settings";

export const componentKeys = {
  all: ["components"] as const,
  selectors: () => [...componentKeys.all, "selectors"] as const,
  valueFunctions: () => [...componentKeys.all, "value-functions"] as const,
  feedbackGenerators: () => [...componentKeys.all, "feedback-generators"] as const,
  artifactHandlers: () => [...componentKeys.all, "artifact-handlers"] as const,
};

export function useSelectors() {
  return useQuery({
    queryKey: componentKeys.selectors(),
    queryFn: () => settingsApi.getAvailableSelectors(),
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