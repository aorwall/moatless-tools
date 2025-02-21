export function useTrajectoryActions() {
  const { trajectoryId } = useTrajectoryId();
  const store = useTrajectoryStore();

  return {
    toggleNode: (nodeId: number) => store.toggleNode(trajectoryId, nodeId),
    toggleItem: (nodeId: number, itemId: string) => store.toggleItem(trajectoryId, nodeId, itemId),
    resetInstance: () => store.resetInstance(trajectoryId),
    // ... rest of actions
  };
} 