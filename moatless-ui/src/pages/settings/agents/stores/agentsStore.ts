import { create } from 'zustand';
import type { AgentConfig, AgentData } from '../types';

interface AgentsState {
  agents: AgentConfig[];
  supportedModels: string[];
  modelConfigs: Record<string, any>;
  loading: boolean;
  error: string | null;
  fetchAgents: () => Promise<void>;
}

export const useAgentsStore = create<AgentsState>((set) => ({
  agents: [],
  supportedModels: [],
  modelConfigs: {},
  loading: false,
  error: null,
  fetchAgents: async () => {
    set({ loading: true, error: null });
    try {
      const [agentsResponse, modelsResponse] = await Promise.all([
        fetch(`http://localhost:8000/agents`),
        fetch(`http://localhost:8000/models`)
      ]);

      if (!agentsResponse.ok || !modelsResponse.ok) {
        throw new Error('Failed to fetch data');
      }

      const [agentsData, modelsData] = await Promise.all([
        agentsResponse.json(),
        modelsResponse.json()
      ]);

      const modelIds = Object.keys(modelsData.models || {}).sort();

      set({
        agents: Object.values(agentsData.configs || {}),
        supportedModels: modelIds,
        modelConfigs: modelsData.models || {},
        loading: false
      });
    } catch (error) {
      console.error('Error fetching agents:', error);
      set({ error: 'Failed to load agents', loading: false });
    }
  },
})); 