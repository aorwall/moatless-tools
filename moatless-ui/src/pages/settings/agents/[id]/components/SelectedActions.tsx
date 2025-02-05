import { X } from "lucide-react";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { Input } from "@/lib/components/ui/input";
import { useActionStore } from "@/lib/stores/actionStore";
import type { ActionConfig, ActionSchema } from "@/lib/types/agent";

interface SelectedActionsProps {
  actions: ActionConfig[];
  onRemove: (className: string) => void;
  onPropertyChange: (className: string, property: string, value: any) => void;
}

export function SelectedActions({
  actions,
  onRemove,
  onPropertyChange,
}: SelectedActionsProps) {
  const { getActionByTitle } = useActionStore();

  const renderPropertyInput = (
    actionSchema: ActionSchema,
    property: string,
    value: any,
    className: string,
  ) => {
    const propertySchema = actionSchema.properties[property];
    if (!propertySchema) return null;

    const handleChange = (newValue: any) => {
      onPropertyChange(className, property, newValue);
    };

    switch (propertySchema.type) {
      case "integer":
        return (
          <Input
            type="number"
            value={value ?? propertySchema.default}
            onChange={(e) => handleChange(parseInt(e.target.value))}
            className="w-32 h-8"
          />
        );
      case "boolean":
        return (
          <Input
            type="checkbox"
            checked={value ?? propertySchema.default}
            onChange={(e) => handleChange(e.target.checked)}
            className="w-4 h-4"
          />
        );
      case "string":
      default:
        return (
          <Input
            type="text"
            value={value ?? propertySchema.default}
            onChange={(e) => handleChange(e.target.value)}
            className="w-32 h-8"
          />
        );
    }
  };

  const sortedActions = actions.sort((a, b) =>
    a.title.localeCompare(b.title),
  );

  return (
    <div className="flex flex-col h-full border rounded-lg overflow-hidden">
      <div className="flex-none p-4 border-b bg-muted/50">
        <h3 className="font-semibold">Selected Actions</h3>
        <p className="text-sm text-muted-foreground">
          {sortedActions.length} action{sortedActions.length === 1 ? "" : "s"}{" "}
          configured
        </p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4">
          {sortedActions.map((actionConfig) => {
            const actionSchema = getActionByTitle(
              actionConfig.title,
            );
            if (!actionSchema) return null;

            return (
              <div
                key={actionConfig.title}
                className="group border rounded-lg p-4 hover:bg-muted/50 transition-colors mb-4 last:mb-0"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{actionSchema.title}</span>
                  <button
                    onClick={() => onRemove(actionConfig.title)}
                    className="text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <p className="text-sm text-muted-foreground mb-4">
                  {actionSchema.description}
                </p>

                <div className="space-y-3">
                  {Object.entries(actionSchema.properties).map(
                    ([propName, prop]) => (
                      <div key={propName} className="space-y-2">
                        <div className="flex justify-between items-baseline">
                          <label className="text-sm font-medium">
                            {prop.title}
                          </label>
                          {renderPropertyInput(
                            actionSchema,
                            propName,
                            actionConfig.properties[propName],
                            actionConfig.title,
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {prop.description}
                        </p>
                      </div>
                    ),
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
