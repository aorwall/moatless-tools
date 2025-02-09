import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LastUsedStore {
  lastUsedAgent: string;
  lastUsedModel: string;
  setLastUsedAgent: (id: string) => void;
  setLastUsedModel: (id: string) => void;
}

export const useLastUsedStore = create<LastUsedStore>()(
  persist(
    (set) => ({
      lastUsedAgent: "",
      lastUsedModel: "",
      setLastUsedAgent: (id) => set({ lastUsedAgent: id }),
      setLastUsedModel: (id) => set({ lastUsedModel: id }),
    }),
    {
      name: "last-used-store",
    },
  ),
); 