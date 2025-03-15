import { create } from "zustand";
import { ActionSchema, ActionSchemaWithClass } from "@/lib/types/agent";
import { agentsApi } from "@/lib/api/agents";

interface ActionStore {
  actions: Record<string, ActionSchema>;
  isLoading: boolean;
  error: Error | null;
  hasLoaded: boolean;
  searchActions: (query: string) => ActionSchemaWithClass[];
  getActionByTitle: (title: string) => ActionSchemaWithClass | undefined;
  getActionByClass: (actionClass: string) => ActionSchemaWithClass | undefined;
  fetchActions: () => Promise<void>;
}

export const useActionStore = create<ActionStore>((set, get) => ({
  actions: {},
  isLoading: false,
  error: null,
  hasLoaded: false,

  searchActions: (query: string) => {
    const { actions } = get();
    const lowercaseQuery = query.toLowerCase();

    return Object.entries(actions).map(([actionClass, action]) => ({
      ...action,
      action_class: actionClass
    })).filter(
      (action) =>
        action.title.toLowerCase().includes(lowercaseQuery) ||
        action.description.toLowerCase().includes(lowercaseQuery),
    );
  },

  getActionByTitle: (title: string) => {
    const { actions } = get();
    const found = Object.entries(actions).find(([_, action]) => action.title === title);
    if (found) {
      const [actionClass, schema] = found;
      return {
        ...schema,
        action_class: actionClass
      };
    }
    return undefined;
  },

  getActionByClass: (actionClass: string) => {
    const { actions } = get();
    const schema = actions[actionClass];
    if (schema) {
      return {
        ...schema,
        action_class: actionClass
      };
    }
    return undefined;
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
