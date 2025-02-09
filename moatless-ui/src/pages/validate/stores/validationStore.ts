import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ValidationStore {
  lastUsedInstance: string;
  lastSearchQuery: string;
  setLastUsedInstance: (id: string) => void;
  setLastSearchQuery: (query: string) => void;
}

export const useValidationStore = create<ValidationStore>()(
  persist(
    (set) => ({
      lastUsedInstance: "",
      lastSearchQuery: "",
      setLastUsedInstance: (id) => set({ lastUsedInstance: id }),
      setLastSearchQuery: (query) => set({ lastSearchQuery: query }),
    }),
    {
      name: "validation-store",
    },
  ),
);
