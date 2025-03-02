import { Control, useWatch } from "react-hook-form";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
  FormMessage,
} from "@/lib/components/ui/form";
import { Loader2, X, ChevronDown, ChevronUp } from "lucide-react";
import {
  ComponentSchema,
  ComponentProperty,
  FlowConfig,
} from "@/lib/types/flow";
import { Badge } from "@/lib/components/ui/badge";
import { Button } from "@/lib/components/ui/button";
import { Checkbox } from "@/lib/components/ui/checkbox";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/lib/components/ui/card";
import { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/lib/components/ui/popover";
import { ScrollArea } from "@/lib/components/ui/scroll-area";
import { ComponentProperties } from "./ComponentProperties";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/lib/components/ui/collapsible";

type ComponentsResponse = Record<string, ComponentSchema>;

interface ArtifactHandlersSelectProps {
  control: Control<FlowConfig>;
  componentsResponse: ComponentsResponse | undefined;
  label: string;
  description: string;
}

export function ArtifactHandlersSelect({
  control,
  componentsResponse,
  label,
  description,
}: ArtifactHandlersSelectProps) {
  const hasComponents =
    componentsResponse && Object.keys(componentsResponse).length > 0;
  const selectedHandlers =
    useWatch({
      control,
      name: "artifact_handlers",
    }) || [];

  const [open, setOpen] = useState(false);
  const [openHandlers, setOpenHandlers] = useState<Record<string, boolean>>({});

  // Get the display name from the full class name
  const getDisplayName = (fullName: string) => {
    return fullName.split(".").pop() || fullName;
  };

  // Check if a handler is selected
  const isSelected = (handlerClass: string) => {
    if (!selectedHandlers || !Array.isArray(selectedHandlers)) return false;
    return selectedHandlers.some(
      (handler: any) =>
        typeof handler === "object" &&
        handler.artifact_handler_class === handlerClass,
    );
  };

  // Toggle the expanded state of a handler's properties
  const toggleHandler = (handlerClass: string) => {
    setOpenHandlers((prev) => ({
      ...prev,
      [handlerClass]: !prev[handlerClass],
    }));
  };

  return (
    <FormField
      control={control}
      name="artifact_handlers"
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <div className="space-y-2">
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className="w-full justify-between"
                    disabled={!hasComponents}
                  >
                    {!componentsResponse ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <span>
                        {selectedHandlers.length > 0
                          ? `${selectedHandlers.length} handler${selectedHandlers.length > 1 ? "s" : ""} selected`
                          : `Select ${label.toLowerCase()}`}
                      </span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-full p-0" align="start">
                  <ScrollArea className="h-72">
                    <div className="p-2 space-y-2">
                      {componentsResponse &&
                        Object.entries(componentsResponse).map(
                          ([key, schema]) => {
                            const displayName = getDisplayName(key);
                            const checked = isSelected(key);

                            return (
                              <div
                                key={key}
                                className="flex items-center space-x-2 p-2 hover:bg-muted rounded-md"
                              >
                                <Checkbox
                                  checked={checked}
                                  onCheckedChange={(isChecked) => {
                                    if (isChecked) {
                                      // Add handler to the list with default properties
                                      const defaults = Object.fromEntries(
                                        Object.entries(
                                          schema?.properties || {},
                                        ).map(([propKey, prop]) => [
                                          propKey,
                                          (prop as ComponentProperty).default,
                                        ]),
                                      );

                                      field.onChange([
                                        ...selectedHandlers,
                                        {
                                          artifact_handler_class: key,
                                          ...defaults,
                                        },
                                      ]);

                                      // Auto-expand the newly added handler
                                      setOpenHandlers((prev) => ({
                                        ...prev,
                                        [key]: true,
                                      }));
                                    } else {
                                      // Remove handler from the list
                                      field.onChange(
                                        selectedHandlers.filter(
                                          (handler: any) =>
                                            handler.artifact_handler_class !==
                                            key,
                                        ),
                                      );
                                    }
                                  }}
                                />
                                <div className="flex-1">
                                  <div className="font-medium">
                                    {displayName}
                                  </div>
                                  {schema.description && (
                                    <div className="text-xs text-muted-foreground">
                                      {schema.description}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          },
                        )}
                    </div>
                  </ScrollArea>
                </PopoverContent>
              </Popover>

              {/* Display selected handlers with their properties */}
              {selectedHandlers.length > 0 && (
                <div className="space-y-3">
                  {selectedHandlers.map((handler: any, index: number) => {
                    const handlerClass = handler.artifact_handler_class;
                    if (!handlerClass) return null;

                    const schema = componentsResponse?.[handlerClass];
                    if (!schema) return null;

                    const displayName = getDisplayName(handlerClass);
                    const isOpen = openHandlers[handlerClass] || false;

                    return (
                      <Card key={index}>
                        <CardHeader className="p-3 pb-0">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-sm font-medium">
                              {displayName}
                            </CardTitle>
                            <div className="flex items-center space-x-1">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6"
                                onClick={() => toggleHandler(handlerClass)}
                              >
                                {isOpen ? (
                                  <ChevronUp className="h-4 w-4" />
                                ) : (
                                  <ChevronDown className="h-4 w-4" />
                                )}
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-destructive"
                                onClick={() => {
                                  field.onChange(
                                    selectedHandlers.filter(
                                      (_: any, i: number) => i !== index,
                                    ),
                                  );
                                }}
                              >
                                <X className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        </CardHeader>
                        <Collapsible
                          open={isOpen}
                          onOpenChange={() => toggleHandler(handlerClass)}
                        >
                          <CollapsibleContent>
                            <CardContent className="p-3 pt-2">
                              {/* Component properties */}
                              <ComponentProperties
                                schema={schema}
                                control={control}
                                basePath={`artifact_handlers.${index}`}
                              />
                            </CardContent>
                          </CollapsibleContent>
                        </Collapsible>
                      </Card>
                    );
                  })}
                </div>
              )}
            </div>
          </FormControl>
          <FormDescription>
            {hasComponents
              ? description
              : `No ${label.toLowerCase()} components are currently available`}
          </FormDescription>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
