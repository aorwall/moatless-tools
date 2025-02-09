import { TrajectoryEvent } from "@/lib/types/trajectory";
import { create } from "zustand";

interface TrajectoryState {
  trajectoryId: string | null;
  expandedNodes: Record<string, Set<number>>;
  expandedItems: Record<string, Record<number, Set<string>>>;
  events: TrajectoryEvent[];
  selectedItem: {
    instanceId: string;
    nodeId: number;
    itemId: string;
    type: string;
    content: any;
  } | null;

  // Actions
  toggleNode: (instanceId: string, nodeId: number) => void;
  toggleItem: (instanceId: string, nodeId: number, itemId: string) => void;
  resetInstance: (instanceId: string) => void;
  setSelectedItem: (item: TrajectoryState["selectedItem"]) => void;
  setTrajectoryId: (trajectoryId: string) => void;
  setEvents: (events: TrajectoryEvent[]) => void;
  addEvent: (event: TrajectoryEvent) => void;

  // State getters
  isNodeExpanded: (instanceId: string, nodeId: number) => boolean;
  isItemExpanded: (
    instanceId: string,
    nodeId: number,
    itemId: string,
  ) => boolean;

}

export const useTrajectoryStore = create<TrajectoryState>((set, get) => ({
  trajectoryId: null,
  expandedNodes: {},
  expandedItems: {},
  selectedItem: null,
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

  setSelectedItem: (item) => set({ selectedItem: item }),

  setTrajectoryId: (trajectoryId: string) => 
    set((state) => {
      // If instance is different, reset the state for the new instance
      if (state.trajectoryId !== trajectoryId) {
        return {
          trajectoryId: trajectoryId,
          expandedNodes: {},
          expandedItems: {},
          selectedItem: null,
        };
      }
      return { trajectoryId: trajectoryId };
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

  addEvent: (event: TrajectoryEvent) => set((state) => ({ events: [...state.events, event] })),
}));
