import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ValidationStore {
  lastUsedAgent: string;
  lastUsedModel: string;
  lastUsedInstance: string;
  lastSearchQuery: string;
  setLastUsedAgent: (id: string) => void;
  setLastUsedModel: (id: string) => void;
  setLastUsedInstance: (id: string) => void;
  setLastSearchQuery: (query: string) => void;
}

export const useValidationStore = create<ValidationStore>()(
  persist(
    (set) => ({
      lastUsedAgent: '',
      lastUsedModel: '',
      lastUsedInstance: '',
      lastSearchQuery: '',
      setLastUsedAgent: (id) => set({ lastUsedAgent: id }),
      setLastUsedModel: (id) => set({ lastUsedModel: id }),
      setLastUsedInstance: (id) => set({ lastUsedInstance: id }),
      setLastSearchQuery: (query) => set({ lastSearchQuery: query }),
    }),
    {
      name: 'validation-store',
    }
  )
); 