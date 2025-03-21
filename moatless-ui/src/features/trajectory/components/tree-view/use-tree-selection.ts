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
        node_id: nodeItem.node_id,
        type: "node",
        parent_id: nodeItem.parent_node_id,
      })
    } else if (item.type === "completion") {
      const completionItem = item as CompletionItem
      setSelectedItem({
        id: item.id,
        node_id: completionItem.node_id,
        type: "completion",
        parent_id: completionItem.parent_id,
      })
    } else if (item.type === "thought") {
      setSelectedItem({
        id: item.id,
        node_id: (item as ThoughtItem).node_id,
        type: "thought",
      })
    } else if (item.type === "action") {
      setSelectedItem({
        id: item.id,
        node_id: (item as ActionItem).node_id,
        type: "action",
      })
    }
  }

  return { selectedItem, selectItem }
}

