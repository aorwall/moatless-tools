import type { TreeItem, NodeItem, ActionItem } from "@/features/trajectory/components/tree-view/types"

// Find an item by its ID
export const findItemById = (id: string, treeData: TreeItem[]): TreeItem | null => {
  // Recursive function to search through all items
  const searchRecursively = (items: TreeItem[]): TreeItem | null => {
    for (const item of items) {
      if (item.id === id) {
        return item
      }

      // Search in children if they exist
      if (item.type === "node" && (item as NodeItem).children) {
        const found = searchRecursively((item as NodeItem).children!)
        if (found) return found
      }

      if (item.type === "action" && (item as ActionItem).children) {
        const found = searchRecursively((item as ActionItem).children!)
        if (found) return found
      }
    }
    return null
  }

  return searchRecursively(treeData)
}

// Check if an item has children
export const hasChildren = (item: TreeItem): boolean => {
  return (
    (item.type === "node" && (item as NodeItem).children && (item as NodeItem).children!.length > 0) ||
    (item.type === "action" && (item as ActionItem).children && (item as ActionItem).children!.length > 0)
  )
}

