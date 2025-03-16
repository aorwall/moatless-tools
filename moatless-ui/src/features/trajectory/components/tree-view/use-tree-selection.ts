"use client"

import { useState } from "react"
import type { TreeItem, SelectedItem, NodeItem, CompletionItem, ThoughtItem, ActionItem } from "@/features/trajectory/components/tree-view/types"

export function useTreeSelection() {
  const [selectedItem, setSelectedItem] = useState<SelectedItem | null>(null)

  const selectItem = (item: TreeItem) => {
    if (item.type === "node") {
      const nodeItem = item as NodeItem
      setSelectedItem({
        id: item.id,
        nodeId: item.id,
        type: "node",
        parentId: nodeItem.parentNodeId,
      })
    } else if (item.type === "completion") {
      const completionItem = item as CompletionItem
      setSelectedItem({
        id: item.id,
        nodeId: completionItem.nodeId,
        type: "completion",
        parentId: completionItem.parentId,
      })
    } else if (item.type === "thought") {
      setSelectedItem({
        id: item.id,
        nodeId: (item as ThoughtItem).nodeId,
        type: "thought",
      })
    } else if (item.type === "action") {
      setSelectedItem({
        id: item.id,
        nodeId: (item as ActionItem).nodeId,
        type: "action",
      })
    }
  }

  return { selectedItem, selectItem }
}

