import { useState, useMemo } from "react";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { Button } from "@/lib/components/ui/button";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface FilterField {
  name: string;
  type: "text" | "select";
  options?: string[];
}

interface ItemAction<T> {
  label: string;
  icon: LucideIcon;
  onClick: (item: T) => void;
}

interface ItemDisplay {
  title: string;
  subtitle?: string;
}

interface DataExplorerProps<T> {
  items: T[];
  filterFields?: FilterField[];
  itemDisplay: (item: T) => ItemDisplay;
  onSelect: (item: T) => void;
  selectedItem?: T;
  itemActions?: ItemAction<T>[];
}

export function DataExplorer<T>({
  items,
  filterFields,
  itemDisplay,
  onSelect,
  selectedItem,
  itemActions,
}: DataExplorerProps<T>) {
  const [filters, setFilters] = useState<Record<string, string>>({});

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      return Object.entries(filters).every(([field, value]) => {
        if (!value) return true;
        const itemValue = (item as any)[field]?.toString().toLowerCase();
        return itemValue?.includes(value.toLowerCase());
      });
    });
  }, [items, filters]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Filters */}
      {filterFields && (
        <div className="flex-none border-b bg-gray-50/50 px-3 py-3">
          <div className="space-y-2">
            {filterFields?.map((field) => (
              <div key={field.name}>
                {field.type === "text" ? (
                  <Input
                    type="text"
                    placeholder={`Search ${field.name}...`}
                    value={filters[field.name] || ""}
                    onChange={(e) =>
                      setFilters((prev) => ({
                        ...prev,
                        [field.name]: e.target.value,
                      }))
                    }
                    className="w-full"
                  />
                ) : field.type === "select" && field.options ? (
                  <Select
                    value={filters[field.name] || "all"}
                    onValueChange={(value) =>
                      setFilters((prev) => ({
                        ...prev,
                        [field.name]: value,
                      }))
                    }
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder={`Filter by ${field.name}`} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All {field.name}s</SelectItem>
                      {field.options.map((option) => (
                        <SelectItem key={option} value={option}>
                          {option}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Item List */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {filteredItems.length > 0 ? (
          filteredItems.map((item, index) => {
            const display = itemDisplay(item);
            const isSelected = selectedItem === item;

            return (
              <div
                key={index}
                className={`p-4 border-b cursor-pointer hover:bg-gray-50 ${
                  isSelected ? "bg-gray-50" : ""
                }`}
                onClick={() => onSelect(item)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium">{display.title}</div>
                    {display.subtitle && (
                      <div className="text-sm text-gray-500">
                        {display.subtitle}
                      </div>
                    )}
                  </div>
                  {itemActions && (
                    <div className="flex gap-2">
                      {itemActions.map((action, actionIndex) => {
                        const Icon = action.icon;
                        return (
                          <Button
                            key={actionIndex}
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              action.onClick(item);
                            }}
                          >
                            <Icon className="h-4 w-4 mr-2" />
                            {action.label}
                          </Button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        ) : (
          <div className="p-4 text-center text-gray-500">No items found</div>
        )}
      </div>
    </div>
  );
}
