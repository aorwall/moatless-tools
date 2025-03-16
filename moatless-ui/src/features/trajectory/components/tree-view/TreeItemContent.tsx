import { TreeIcon } from "@/features/trajectory/components/tree-view/TreeIcon"
import type { TreeItem, NodeItem, CompletionItem } from "@/features/trajectory/components/tree-view/types"

interface TreeItemContentProps {
  item: TreeItem
}

export function TreeItemContent({ item }: TreeItemContentProps) {
  return (
    <div className="flex items-center">
      <TreeIcon item={item} />
      <span className="font-medium ml-2">{item.label}</span>
      {item.type === "node" && <span className="text-gray-500 ml-2 text-xs">{(item as NodeItem).timestamp}</span>}
      {item.detail && <span className="text-gray-600 ml-1">{item.detail}</span>}
      {item.time && <span className="text-gray-500 ml-2 text-xs">{item.time}</span>}
      {item.type === "completion" && (item as CompletionItem).tokens && (
        <span className="ml-2 px-1.5 py-0.5 text-xs bg-gray-200 text-gray-700 rounded-full">
          {(item as CompletionItem).tokens} tokens
        </span>
      )}
    </div>
  )
}

