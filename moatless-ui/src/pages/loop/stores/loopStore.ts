import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LoopStore {
  lastUsedAgent: string;
  lastUsedModel: string;
  lastMessage: string;
  setLastUsedAgent: (id: string) => void;
  setLastUsedModel: (id: string) => void;
  setLastMessage: (message: string) => void;
}

export const useLoopStore = create<LoopStore>()(
  persist(
    (set) => ({
      lastUsedAgent: "",
      lastUsedModel: "",
      lastMessage: "",
      setLastUsedAgent: (id) => set({ lastUsedAgent: id }),
      setLastUsedModel: (id) => set({ lastUsedModel: id }),
      setLastMessage: (message) => set({ lastMessage: message }),
    }),
    {
      name: "loop-store",
    },
  ),
);
