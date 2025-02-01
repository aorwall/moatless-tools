import { create } from 'zustand';
import type { ModelConfig } from '../../../../lib/types/model';

interface ModelsState {
  models: ModelConfig[];
  loading: boolean;
  error: string | null;
}

interface ModelsActions {
  fetchModels: () => Promise<void>;
  setError: (error: string | null) => void;
  reset: () => void;
}

const initialState: ModelsState = {
  models: [],
  loading: false,
  error: null,
};

export const useModelsStore = create<ModelsState & ModelsActions>()((set) => ({
  ...initialState,

  reset: () => set(initialState),

  setError: (error) => set({ error }),

  fetchModels: async () => {
    try {
      set({ loading: true, error: null });
      const response = await fetch(`http://localhost:8000/api/models`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      set({ models: data.models, loading: false });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load models';
      set({ error: errorMessage, loading: false });
    }
  },
}));

// Optional: Create hooks for specific use cases
export const useModels = () => useModelsStore((state) => state.models);
export const useModelsLoading = () => useModelsStore((state) => state.loading);
export const useModelsError = () => useModelsStore((state) => state.error); 