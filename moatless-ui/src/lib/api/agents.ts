import type {
  AgentConfig,
  AgentData,
  ActionsResponse,
  ActionSchema,
} from "@/lib/types/agent";
import { apiRequest } from "./config";

export const agentsApi = {
  getAgents: () => apiRequest<AgentData>("/agents"),
  getAgent: (id: string) => apiRequest<AgentConfig>(`/agents/${id}`),
  updateAgent: (agent: AgentConfig) =>
    apiRequest<AgentConfig>(`/agents/${agent.id}`, {
      method: "PUT",
      body: JSON.stringify(agent),
    }),
  getAvailableActions: () =>
    apiRequest<ActionSchema[]>("/settings/components/actions"),
  deleteAgent: (id: string) =>
    apiRequest<void>(`/agents/${id}`, {
      method: "DELETE",
    }),
  createAgent: (agent: Omit<AgentConfig, "id">) =>
    apiRequest<AgentConfig>("/agents", {
      method: "POST",
      body: JSON.stringify(agent),
    }),
};
