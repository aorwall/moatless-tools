"use client"

import React, { useState, useEffect } from 'react';
import { TreeItem as TreeItemComponent } from './TreeItem';
import { NodeItem, TreeItem } from './types';
import { useTrajectoryStore } from '../../stores/trajectoryStore';
import { Trajectory } from '@/lib/types/trajectory';

interface TreeViewProps {
  trajectory: Trajectory;
  treeData: NodeItem;
  loading: boolean;
  error: Error | null;
}

const TreeView: React.FC<TreeViewProps> = ({
  trajectory,
  treeData,
  loading,
  error
}) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const { setSelectedNode, setSelectedTreeItem, getSelectedTreeItem } = useTrajectoryStore();

  // Initialize all items as expanded when treeData changes
  useEffect(() => {
    if (treeData) {
      const initialExpandedState: Record<string, boolean> = {};
      initialExpandedState[treeData.id] = true;

      // Helper function to recursively add all item IDs to the expanded state
      const addItemsToExpandedState = (items: TreeItem[]) => {
        items.forEach(item => {
          initialExpandedState[item.id] = true;
          if ('children' in item && item.children && item.children.length > 0) {
            addItemsToExpandedState(item.children);
          }
        });
      };

      addItemsToExpandedState(treeData.children || []);
      setExpanded(initialExpandedState);
    }
  }, [treeData]);

  const toggleExpand = (id: string) => {
    setExpanded((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const isItemSelected = (item: TreeItem) => {
    return getSelectedTreeItem(trajectory.trajectory_id)?.id === item.id;
  };

  const handleSelectItem = (item: TreeItem) => {
    setSelectedTreeItem(item);
    setSelectedNode(trajectory.trajectory_id, item.node_id);
  };

  if (loading && !treeData) {
    return <div className="flex justify-center items-center h-24 text-muted-foreground italic">Loading tree view data...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-24 text-destructive italic">Error: {error.message}</div>;
  }

  if (!treeData) {
    return <div className="flex justify-center items-center h-24 text-muted-foreground italic">No tree data available</div>;
  }

  return (
    <div className="space-y-1 p-1 pt-2 text-sm overflow-y-auto max-h-[80vh]">
      <TreeItemComponent
        key={treeData.id}
        item={treeData}
        level={0}
        expanded={expanded}
        toggleExpand={toggleExpand}
        isSelected={isItemSelected}
        onSelect={handleSelectItem}
      />
    </div>
  );
};

export default TreeView;

