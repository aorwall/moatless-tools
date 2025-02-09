import { useState } from "react";
import { Input } from "@/lib/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { cn } from "@/lib/utils";

interface FilterField {
  name: string;
  type: "text" | "select";
  options?: any[];
}

interface ItemDisplay {
  title: string;
  subtitle: string;
}

interface DataExplorerProps {
  items: any[];
  filterFields: FilterField[];
  itemDisplay: (item: any) => ItemDisplay;
  onSelect: (item: any) => void;
  selectedItem?: any;
  compareItems?: (a: any, b: any) => boolean;
}

export function DataExplorer({
  items,
  filterFields,
  itemDisplay,
  onSelect,
  selectedItem,
  compareItems = (a, b) => a === b,
}: DataExplorerProps) {
  const [filters, setFilters] = useState<Record<string, any>>({});

  const filteredItems = items.filter((item) => {
    return filterFields.every((field) => {
      const filterValue = filters[field.name];
      if (!filterValue || filterValue === "all") return true;

      if (field.type === "select") {
        return item[field.name] === filterValue;
      } else {
        const itemValue = item[field.name]?.toString().toLowerCase() || "";
        return itemValue.includes(filterValue.toLowerCase());
      }
    });
  });

  const handleFilterChange = (fieldName: string, value: any) => {
    setFilters((prev) => ({
      ...prev,
      [fieldName]: value,
    }));
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Filters */}
      <div className="flex-none border-b bg-gray-50/50 px-3 py-3">
        <div className="space-y-2">
          {filterFields.map((field) => (
            <div key={field.name}>
              {field.type === "text" ? (
                <Input
                  type="text"
                  placeholder={`Search ${field.name}...`}
                  value={filters[field.name] || ""}
                  onChange={(e) =>
                    handleFilterChange(field.name, e.target.value)
                  }
                  className="w-full"
                />
              ) : field.type === "select" && field.options ? (
                <Select
                  value={filters[field.name] || "all"}
                  onValueChange={(value) =>
                    handleFilterChange(field.name, value)
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

      {/* Item List */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {filteredItems.length > 0 ? (
          filteredItems.map((item) => {
            const display = itemDisplay(item);
            return (
              <button
                key={display.title}
                className={cn(
                  "w-full border-b px-4 py-3 text-left transition-colors hover:bg-gray-50 focus:bg-gray-50 focus:outline-none",
                  selectedItem &&
                    compareItems(selectedItem, item) &&
                    "bg-blue-50 hover:bg-blue-50",
                )}
                onClick={() => onSelect(item)}
              >
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-medium">{display.title}</div>
                    <div className="mt-0.5 text-sm text-gray-500">
                      {display.subtitle}
                    </div>
                  </div>
                </div>
              </button>
            );
          })
        ) : (
          <div className="p-4 text-center text-gray-500">No items found</div>
        )}
      </div>
    </div>
  );
}
