import { TrajectoryEvent } from "@/lib/types/trajectory.ts";
import { create } from "zustand";

interface TrajectoryState {
  expandedNodes: Record<string, Set<number>>;
  expandedItems: Record<string, Record<number, Set<string>>>;
  events: TrajectoryEvent[];
  selectedItems: Record<
    string,
    {
      nodeId: number;
      itemId: string | null;
      type: string;
      content: any;
    }
  >;

  // Actions
  toggleNode: (instanceId: string, nodeId: number) => void;
  toggleItem: (instanceId: string, nodeId: number, itemId: string) => void;
  resetInstance: (instanceId: string) => void;
  setSelectedItem: (
    instanceId: string,
    item: Omit<
      NonNullable<TrajectoryState["selectedItems"][string]>,
      "instanceId"
    > | null,
  ) => void;
  setSelectedNode: (instanceId: string, nodeId: number) => void;

  setEvents: (events: TrajectoryEvent[]) => void;
  addEvent: (event: TrajectoryEvent) => void;

  // State getters
  isNodeExpanded: (instanceId: string, nodeId: number) => boolean;
  isItemExpanded: (
    instanceId: string,
    nodeId: number,
    itemId: string,
  ) => boolean;
  getSelectedItem: (
    instanceId: string,
  ) => TrajectoryState["selectedItems"][string] | null;
}

export const useTrajectoryStore = create<TrajectoryState>((set, get) => ({
  expandedNodes: {},
  expandedItems: {},
  selectedItems: {},
  events: [],

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

  setSelectedItem: (instanceId, item) =>
    set((state) => {
      const newSelectedItems = { ...state.selectedItems };
      if (item === null) {
        delete newSelectedItems[instanceId];
      } else {
        newSelectedItems[instanceId] = item;
      }
      return { selectedItems: newSelectedItems };
    }),

  getSelectedItem: (instanceId) => {
    const state = get();
    return state.selectedItems[instanceId] || null;
  },

  setSelectedNode: (instanceId, nodeId) =>
    set((state) => {
      const newSelectedItems = { ...state.selectedItems };
      newSelectedItems[instanceId] = { nodeId, itemId: null, type: "node", content: null };
      return { selectedItems: newSelectedItems };
    }),

  isNodeExpanded: (instanceId, nodeId) => {
    const state = get();
    return state.expandedNodes[instanceId]?.has(nodeId) || false;
  },

  isItemExpanded: (instanceId, nodeId, itemId) => {
    const state = get();
    return state.expandedItems[instanceId]?.[nodeId]?.has(itemId) || false;
  },

  setEvents: (events: TrajectoryEvent[]) => set({ events }),

  addEvent: (event: TrajectoryEvent) =>
    set((state) => ({ events: [...state.events, event] })),
}));
