import { useState } from "react"
import { ChevronDown, Plus, X } from "lucide-react"
import type { DynamicListField, DynamicListItem } from "@/lib/components/form/types"
import { Button } from "@/lib/components/ui/button"
import { DynamicField } from "@/lib/components/form/dynamic-field"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/lib/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { ItemSelectorDialog } from "@/lib/components/form/item-selector-dialog"

interface DynamicItemListProps {
  field: DynamicListField
  value: DynamicListItem[]
  onChange: (id: string, value: DynamicListItem[]) => void
}

export function DynamicItemList({ field, value = [], onChange }: DynamicItemListProps) {
  const [openStates, setOpenStates] = useState<Record<string, boolean>>({})
  const [selectorOpen, setSelectorOpen] = useState(false)

  // Filter out already selected items
  const availableItems = field.availableItems.filter((item) => !value.some((v) => v.id === item.id))

  // Sort selected items by name
  const sortedValue = [...value].sort((a, b) => a.name.localeCompare(b.name))

  const toggleItem = (itemId: string) => {
    setOpenStates((prev) => ({
      ...prev,
      [itemId]: !prev[itemId],
    }))
  }

  const addItem = (item: DynamicListItem) => {
    onChange(field.id, [...value, { ...item, values: {} }])
    // Auto-open the newly added item
    setOpenStates((prev) => ({
      ...prev,
      [item.id]: true,
    }))
  }

  const removeItem = (itemId: string) => {
    onChange(
      field.id,
      value.filter((item) => item.id !== itemId),
    )
  }

  const updateItemField = (itemId: string, fieldId: string, fieldValue: any) => {
    onChange(
      field.id,
      value.map((item) =>
        item.id === itemId
          ? {
            ...item,
            values: { ...item.values, [fieldId]: fieldValue },
          }
          : item,
      ),
    )
  }

  // Default button text or use custom text if provided
  const addButtonText = field.addButtonText || "Add Item"

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium">{field.label}</h3>
          <p className="text-sm text-muted-foreground">{value.length} items configured</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSelectorOpen(true)}
          disabled={availableItems.length === 0}
        >
          <Plus className="w-4 h-4 mr-2" />
          {addButtonText}
        </Button>
      </div>

      <div className="space-y-2">
        {sortedValue.map((item) => (
          <Collapsible
            key={item.id}
            open={openStates[item.id]}
            onOpenChange={() => toggleItem(item.id)}
            className="border rounded-lg"
          >
            <CollapsibleTrigger className="flex items-center justify-between w-full p-4 hover:bg-muted/50">
              <div className="flex flex-col items-start gap-1">
                <div className="font-medium">{item.name}</div>
                <div className="text-sm text-muted-foreground">{item.description}</div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeItem(item.id)
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
                <ChevronDown className={cn("w-4 h-4 transition-transform", openStates[item.id] && "rotate-180")} />
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="p-4 pt-0 space-y-4">
              {item.fields.map((field) => (
                <DynamicField
                  key={field.id}
                  field={field}
                  value={item.values?.[field.id]}
                  onChange={(fieldId, fieldValue) => updateItemField(item.id, fieldId, fieldValue)}
                />
              ))}
            </CollapsibleContent>
          </Collapsible>
        ))}

        {value.length === 0 && (
          <div className="text-center py-8 border rounded-lg border-dashed">
            <p className="text-muted-foreground">No items added yet</p>
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() => setSelectorOpen(true)}
              disabled={availableItems.length === 0}
            >
              <Plus className="w-4 h-4 mr-2" />
              {addButtonText}
            </Button>
          </div>
        )}
      </div>

      <ItemSelectorDialog
        title={`Select ${field.addButtonText?.replace("Add ", "") || "Item"}`}
        items={availableItems}
        open={selectorOpen}
        onOpenChange={setSelectorOpen}
        onSelect={addItem}
      />
    </div>
  )
}

