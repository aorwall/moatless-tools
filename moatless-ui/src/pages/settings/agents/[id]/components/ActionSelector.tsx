import { useState, useEffect } from "react";
import { Search } from "lucide-react";
import { Input } from "@/lib/components/ui/input";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Button } from "@/lib/components/ui/button";
import { useActionStore } from "@/lib/stores/actionStore";
import type { ActionSchema } from "@/lib/types/agent";

interface ActionSelectorProps {
  selectedActions: string[];
  onSelect: (actionClassName: string) => void;
}

export function ActionSelector({
  selectedActions,
  onSelect,
}: ActionSelectorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const { actions, isLoading, error, fetchActions, searchActions } =
    useActionStore();

  useEffect(() => {
    fetchActions();
  }, [fetchActions]);

  const filteredActions = Object.entries(actions).filter(([_, action]) => {
    const query = searchQuery.toLowerCase();
    const isSelected = Object.values(selectedActions).some(
      (selected) => selected === action.title,
    );
    return (
      !isSelected &&
      (action.title.toLowerCase().includes(query) ||
        action.description.toLowerCase().includes(query))
    );
  });

  const categories = groupActionsByCategory(
    Object.fromEntries(filteredActions),
  );

  function groupActionsByCategory(actions: Record<string, ActionSchema>) {
    return Object.entries(actions).reduce(
      (acc, [key, action]) => {
        const category = action.title.split("_")[0] || "Other";
        if (!acc[category]) {
          acc[category] = {};
        }
        acc[category][key] = action;
        return acc;
      },
      {} as Record<string, Record<string, ActionSchema>>,
    );
  }

  if (error) {
    return <div className="text-destructive">Failed to load actions</div>;
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-32">
        Loading actions...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full border rounded-lg overflow-hidden">
      <div className="flex-none p-4 border-b bg-muted/50">
        <h3 className="font-semibold">Add Actions</h3>
        <div className="relative mt-2">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search available actions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {Object.entries(categories).map(([category, categoryActions]) => (
            <div key={category} className="mb-6 last:mb-0">
              <h3 className="font-semibold mb-3 text-sm text-muted-foreground">
                {category}
              </h3>
              <div className="space-y-2">
                {Object.entries(categoryActions).map(([key, action]) => (
                  <div
                    key={action.title}
                    className="flex flex-col space-y-2 p-3 rounded-lg hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="font-medium">{action.title}</div>
                        <p className="text-sm text-muted-foreground">
                          {action.description}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onSelect(action.title)}
                      >
                        Add
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
