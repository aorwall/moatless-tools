import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LastUsedStore {
  lastUsedModel: string;
  lastUsedFlow: string;
  setLastUsedModel: (id: string) => void;
  setLastUsedFlow: (id: string) => void;
}

export const useLastUsedStore = create<LastUsedStore>()(
  persist(
    (set) => ({
      lastUsedModel: "",
      lastUsedFlow: "",
      setLastUsedModel: (id: string) => set({ lastUsedModel: id }),
      setLastUsedFlow: (id: string) => set({ lastUsedFlow: id }),
    }),
    {
      name: "last-used-store",
    },
  ),
); 