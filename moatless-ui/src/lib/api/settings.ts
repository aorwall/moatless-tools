import { apiRequest } from "./config";
import type { FlowConfig, ComponentSchema } from "@/lib/types/flow";

type FlowsResponse = {
  flows: FlowConfig[];
};

type ComponentsResponse = Record<string, ComponentSchema>;

export const settingsApi = {
  getFlows: () => apiRequest<FlowsResponse>("/settings/flows"),

  getFLow: (id: string) => apiRequest<FlowConfig>(`/settings/flows/${id}`),

  createFlow: (config: FlowConfig) =>
    apiRequest<FlowConfig>("/settings/flows", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  updateFlow: (id: string, config: FlowConfig) =>
    apiRequest<FlowConfig>(`/settings/flows/${id}`, {
      method: "PUT",
      body: JSON.stringify(config),
    }),

  // Component endpoints
  getAvailableSelectors: () =>
    apiRequest<ComponentsResponse>("/settings/components/selectors"),

  getAvailableValueFunctions: () =>
    apiRequest<ComponentsResponse>("/settings/components/value-functions"),

  getAvailableFeedbackGenerators: () =>
    apiRequest<ComponentsResponse>("/settings/components/feedback-generators"),

  getAvailableArtifactHandlers: () =>
    apiRequest<ComponentsResponse>("/settings/components/artifact-handlers"),

  getAvailableMemory: () =>
    apiRequest<ComponentsResponse>("/settings/components/memory"),

  // Generic component endpoint
  getComponentsByType: (componentType: string) =>
    apiRequest<ComponentsResponse>(`/settings/components/${componentType}`),
};
