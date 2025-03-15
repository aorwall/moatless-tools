"use client"

import { useState } from "react"
import { Search, X } from "lucide-react"
import type { DynamicListItem } from "@/lib/components/form/types"
import { Button } from "@/lib/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/lib/components/ui/dialog"
import { Input } from "@/lib/components/ui/input"
import { ScrollArea } from "@/lib/components/ui/scroll-area"
import { cn } from "@/lib/utils"

interface ItemSelectorDialogProps {
  title: string
  items: DynamicListItem[]
  open: boolean
  onOpenChange: (open: boolean) => void
  onSelect: (item: DynamicListItem) => void
}

export function ItemSelectorDialog({ title, items, open, onOpenChange, onSelect }: ItemSelectorDialogProps) {
  const [searchQuery, setSearchQuery] = useState("")

  // Sort items by name
  const sortedItems = [...items].sort((a, b) => a.name.localeCompare(b.name))

  // Filter items based on search query
  const filteredItems = sortedItems.filter(
    (item) =>
      item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[700px] max-h-[85vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <div className="relative mt-4">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search items..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <Button
                variant="ghost"
                size="sm"
                className="absolute right-1 top-1 h-7 w-7 p-0"
                onClick={() => setSearchQuery("")}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-[50vh]">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-1">
              {filteredItems.map((item) => (
                <button
                  key={item.id}
                  className={cn(
                    "flex flex-col text-left p-4 rounded-lg border border-border",
                    "hover:border-primary/50 hover:bg-accent transition-colors",
                    "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                  )}
                  onClick={() => {
                    onSelect(item)
                    onOpenChange(false)
                  }}
                >
                  <div className="font-medium text-lg">{item.name}</div>
                  <div className="text-sm text-muted-foreground mt-1">{item.description}</div>
                  <div className="text-xs text-muted-foreground mt-2">
                    {item.fields.length} configuration {item.fields.length === 1 ? "field" : "fields"}
                  </div>
                </button>
              ))}
            </div>

            {filteredItems.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">No items found matching "{searchQuery}"</div>
            )}
          </ScrollArea>
        </div>
      </DialogContent>
    </Dialog>
  )
}

