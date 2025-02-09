import { create } from "zustand";

interface TrajectoryState {
  // Only track expansion states by instanceId
  expandedNodes: Record<string, Set<number>>;
  expandedItems: Record<string, Record<number, Set<string>>>;

  // Actions
  toggleNode: (instanceId: string, nodeId: number) => void;
  toggleItem: (instanceId: string, nodeId: number, itemId: string) => void;
  resetInstance: (instanceId: string) => void;

  // State getters
  isNodeExpanded: (instanceId: string, nodeId: number) => boolean;
  isItemExpanded: (
    instanceId: string,
    nodeId: number,
    itemId: string,
  ) => boolean;
}

export const useTrajectoryStore = create<TrajectoryState>((set, get) => ({
  // State
  expandedNodes: {},
  expandedItems: {},

  // Actions
  toggleNode: (instanceId, nodeId) =>
    set((state) => {
      const currentSet = state.expandedNodes[instanceId] || new Set();
      const newSet = new Set(currentSet);

      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }

      return {
        expandedNodes: {
          ...state.expandedNodes,
          [instanceId]: newSet,
        },
      };
    }),

  toggleItem: (instanceId, nodeId, itemId) =>
    set((state) => {
      const nodeItems = state.expandedItems[instanceId]?.[nodeId] || new Set();
      const newItems = new Set(nodeItems);

      if (newItems.has(itemId)) {
        newItems.delete(itemId);
      } else {
        newItems.add(itemId);
      }

      return {
        expandedItems: {
          ...state.expandedItems,
          [instanceId]: {
            ...state.expandedItems[instanceId],
            [nodeId]: newItems,
          },
        },
      };
    }),

  resetInstance: (instanceId) =>
    set((state) => ({
      expandedNodes: {
        ...state.expandedNodes,
        [instanceId]: new Set(),
      },
      expandedItems: {
        ...state.expandedItems,
        [instanceId]: {},
      },
    })),

  // Getters
  isNodeExpanded: (instanceId, nodeId) => {
    const state = get();
    return state.expandedNodes[instanceId]?.has(nodeId) || false;
  },

  isItemExpanded: (instanceId, nodeId, itemId) => {
    const state = get();
    return state.expandedItems[instanceId]?.[nodeId]?.has(itemId) || false;
  },
}));
