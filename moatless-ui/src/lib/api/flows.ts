import type { FlowConfig } from "@/lib/types/flow";
import { apiRequest } from "./config";

export const flowsApi = {
  getFlows: () => apiRequest<FlowConfig[]>("/settings/flows"),

  getFlow: (id: string) => apiRequest<FlowConfig>(`/settings/flows/${id}`),

  createFlow: (flow: FlowConfig) =>
    apiRequest<FlowConfig>("/settings/flows", {
      method: "POST",
      body: JSON.stringify(flow),
    }),

  updateFlow: (id: string, flow: FlowConfig) =>
    apiRequest<FlowConfig>(`/settings/flows/${id}`, {
      method: "PUT",
      body: JSON.stringify(flow),
    }),

  deleteFlow: (id: string) =>
    apiRequest<void>(`/settings/flows/${id}`, {
      method: "DELETE",
    }),
};
