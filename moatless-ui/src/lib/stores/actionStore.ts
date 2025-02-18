import { create } from "zustand";
import { ActionSchema } from "@/lib/types/agent";
import { agentsApi } from "@/lib/api/agents";

interface ActionStore {
  actions: ActionSchema[];
  isLoading: boolean;
  error: Error | null;
  hasLoaded: boolean;
  searchActions: (query: string) => ActionSchema[];
  getActionByTitle: (title: string) => ActionSchema | undefined;
  fetchActions: () => Promise<void>;
}

export const useActionStore = create<ActionStore>((set, get) => ({
  actions: [],
  isLoading: false,
  error: null,
  hasLoaded: false,

  searchActions: (query: string) => {
    const { actions } = get();
    const lowercaseQuery = query.toLowerCase();

    return Object.values(actions).filter(
      (action) =>
        action.title.toLowerCase().includes(lowercaseQuery) ||
        action.description.toLowerCase().includes(lowercaseQuery),
    );
  },

  getActionByTitle: (title: string) => {
    const { actions } = get();
    return Object.values(actions).find((action) => action.title === title);
  },

  fetchActions: async () => {
    const { hasLoaded, isLoading } = get();

    if (hasLoaded || isLoading) return;

    set({ isLoading: true, error: null });
    try {
      const response = await agentsApi.getAvailableActions();
      set({ actions: response, isLoading: false, hasLoaded: true });
    } catch (error) {
      set({ error: error as Error, isLoading: false });
    }
  },
}));
