import { useTrajectoryContext } from "@/lib/contexts/TrajectoryContext";
import { TrajectoryEvent, Trajectory } from "@/lib/types/trajectory";
import { create } from "zustand";

interface TrajectoryState {
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

// Custom hook to combine store with context
export function useTrajectoryActions() {
  const { trajectory } = useTrajectoryContext();
  const store = useTrajectoryStore();

  return {
    toggleNode: (nodeId: number) => store.toggleNode(trajectory.id, nodeId),
    toggleItem: (nodeId: number, itemId: string) => store.toggleItem(trajectory.id, nodeId, itemId),
    resetInstance: () => store.resetInstance(trajectory.id),
    isNodeExpanded: (nodeId: number) => store.isNodeExpanded(trajectory.id, nodeId),
    isItemExpanded: (nodeId: number, itemId: string) => store.isItemExpanded(trajectory.id, nodeId, itemId),
    setSelectedItem: store.setSelectedItem,
    selectedItem: store.selectedItem,
    events: store.events,
    setEvents: store.setEvents,
    addEvent: store.addEvent,
  };
}
