import { create } from "zustand";
import type { ModelConfig } from "@/lib/types/model";

interface ModelsState {
  models: ModelConfig[];
  loading: boolean;
  error: string | null;
  fetchModels: () => Promise<void>;
}

export const useModelsStore = create<ModelsState>((set) => ({
  models: [],
  loading: false,
  error: null,
  fetchModels: async () => {
    set({ loading: true, error: null });
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`http://localhost:8000/models`, {
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }
      const data = await response.json();
      set({ models: data.models, loading: false });
    } catch (error) {
      console.error("Error fetching models:", error);
      set({
        error: error instanceof Error ? error.message : "Failed to load models",
        loading: false,
      });
    }
  },
}));
