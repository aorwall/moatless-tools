import { create } from "zustand";
import { persist } from "zustand/middleware";

interface RunStore {
  selectedAgentId: string;
  selectedModelId: string;
  message: string;
  repositoryPath: string;
  setSelectedAgentId: (id: string) => void;
  setSelectedModelId: (id: string) => void;
  setMessage: (message: string) => void;
  setRepositoryPath: (path: string) => void;
}

export const useRunStore = create<RunStore>()(
  persist(
    (set) => ({
      selectedAgentId: "",
      selectedModelId: "",
      message: "",
      repositoryPath: "",
      setSelectedAgentId: (id) => set({ selectedAgentId: id }),
      setSelectedModelId: (id) => set({ selectedModelId: id }),
      setMessage: (message) => set({ message: message }),
      setRepositoryPath: (path) => set({ repositoryPath: path }),
    }),
    {
      name: "run-store",
    },
  ),
);
