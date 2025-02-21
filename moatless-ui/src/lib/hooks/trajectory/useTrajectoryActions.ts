import { useTrajectoryId } from '@/lib/contexts/TrajectoryContext';
import { useTrajectoryStore } from '@/lib/stores/trajectoryStore';

export function useTrajectoryActions() {
  const { trajectoryId } = useTrajectoryId();
  const store = useTrajectoryStore();

  console.log("useTrajectoryActions trajectoryId:", trajectoryId);

  if (!trajectoryId) {
    throw new Error('useTrajectoryActions must be used within a TrajectoryProvider');
  }

  return {
    toggleNode: (nodeId: number) => store.toggleNode(trajectoryId, nodeId),
    toggleItem: (nodeId: number, itemId: string) => store.toggleItem(trajectoryId, nodeId, itemId),
    resetInstance: () => store.resetInstance(trajectoryId),
    isNodeExpanded: (nodeId: number) => store.isNodeExpanded(trajectoryId, nodeId),
    isItemExpanded: (nodeId: number, itemId: string) => store.isItemExpanded(trajectoryId, nodeId, itemId),
    setSelectedItem: store.setSelectedItem,
    selectedItem: store.selectedItem,
    events: store.events,
    setEvents: store.setEvents,
    addEvent: store.addEvent,
  };
} 