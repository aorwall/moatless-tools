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
    apiRequest<AgentConfig>(`/agents/${agent.agent_id}`, {
      method: "PUT",
      body: JSON.stringify(agent),
    }),
  getAvailableActions: () =>
    apiRequest<Record<string, ActionSchema>>("/settings/components/actions"),
  deleteAgent: (id: string) =>
    apiRequest<void>(`/agents/${id}`, {
      method: "DELETE",
    }),
  createAgent: (agent: Omit<AgentConfig, "agent_id">) =>
    apiRequest<AgentConfig>("/agents", {
      method: "POST",
      body: JSON.stringify(agent),
    }),
};
