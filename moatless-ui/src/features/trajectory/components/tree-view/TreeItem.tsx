import { ChevronDown, ChevronRight } from "lucide-react"
import { TreeItemContent } from "@/features/trajectory/components/tree-view/TreeItemContent"
import type { TreeItem as TreeItemType, NodeItem, ActionItem } from "@/features/trajectory/components/tree-view/types"
import { hasChildren } from "@/features/trajectory/components/tree-view/utils"

interface TreeItemProps {
  item: TreeItemType
  level: number
  expanded: Record<string, boolean>
  toggleExpand: (id: string) => void
  isSelected: (item: TreeItemType) => boolean
  onSelect: (item: TreeItemType) => void
}

export function TreeItem({ item, level, expanded, toggleExpand, isSelected, onSelect }: TreeItemProps) {
  const itemHasChildren = hasChildren(item)
  const isExpanded = expanded[item.id]
  const selected = isSelected(item)

  return (
    <div>
      <div
        className={`flex items-center py-1 px-2 rounded cursor-pointer ${selected ? "bg-blue-100" : "hover:bg-gray-100"}`}
        style={{ marginLeft: `${level * 16}px` }}
        onClick={() => onSelect(item)}
      >
        {itemHasChildren ? (
          <div
            className="mr-1"
            onClick={(e) => {
              e.stopPropagation()
              toggleExpand(item.id)
            }}
          >
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-500" />
            )}
          </div>
        ) : (
          <div className="w-4 mr-1"></div>
        )}

        <TreeItemContent item={item} />
      </div>

      {itemHasChildren && isExpanded && (
        <div>
          {item.type === "node" &&
            (item as NodeItem).children?.map((child) => (
              <TreeItem
                key={child.id}
                item={child}
                level={level + 1}
                expanded={expanded}
                toggleExpand={toggleExpand}
                isSelected={isSelected}
                onSelect={onSelect}
              />
            ))}
          {item.type === "action" &&
            (item as ActionItem).children?.map((child) => (
              <TreeItem
                key={child.id}
                item={child}
                level={level + 1}
                expanded={expanded}
                toggleExpand={toggleExpand}
                isSelected={isSelected}
                onSelect={onSelect}
              />
            ))}
        </div>
      )}
    </div>
  )
}

