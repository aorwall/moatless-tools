import { Control, useWatch } from "react-hook-form";
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormDescription,
  FormMessage,
} from "@/lib/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/components/ui/select";
import { Loader2 } from "lucide-react";
import { ComponentSchema, ComponentProperty, FlowConfig } from "@/lib/types/flow";
import { ComponentProperties } from "./ComponentProperties";

type ComponentField = "selector" | "value_function" | "feedback_generator";

type ComponentsResponse = Record<string, ComponentSchema>;

interface ComponentSelectProps {
  name: ComponentField;
  control: Control<FlowConfig>;
  componentsResponse: ComponentsResponse | undefined;
  label: string;
  description: string;
}

function getComponentTitle(schema: ComponentSchema): string {
  // The title is either directly in the schema or in the last part of the component name
  return schema.title || schema.$id?.split('.').pop() || 'Unnamed Component';
}

export function ComponentSelect({
  name,
  control,
  componentsResponse,
  label,
  description,
}: ComponentSelectProps) {
  const hasComponents = componentsResponse && Object.keys(componentsResponse).length > 0;
  const selectedValue = useWatch({
    control,
    name,
  });
  const selectedSchema = selectedValue?.[`${name}_class`] 
    ? componentsResponse?.[selectedValue[`${name}_class`]]
    : null;

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <Select 
            onValueChange={(value) => {
              if (value === "none") {
                field.onChange(undefined);
              } else {
                const schema = componentsResponse?.[value];
                const defaults = Object.fromEntries(
                  Object.entries(schema?.properties || {})
                    .map(([key, prop]) => [key, (prop as ComponentProperty).default])
                );
                field.onChange({
                  [`${name}_class`]: value,
                  ...defaults
                });
              }
            }}
            value={field.value?.[`${name}_class`] || "none"}
            disabled={!hasComponents}
          >
            <FormControl>
              <SelectTrigger>
                {!componentsResponse ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SelectValue 
                    placeholder={hasComponents 
                      ? `Select ${label.toLowerCase()}`
                      : `No ${label.toLowerCase()} components available`
                    } 
                  />
                )}
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              {componentsResponse && Object.entries(componentsResponse).map(([key, schema]) => {
                const displayName = key.split('.').pop() || key; // Get the last part of the fully qualified name
                return (
                  <SelectItem key={key} value={key}>
                    {displayName}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          <FormDescription>
            {hasComponents 
              ? description 
              : `No ${label.toLowerCase()} components are currently available`
            }
          </FormDescription>
          <FormMessage />
          
          {/* Show properties if a component is selected */}
          {selectedSchema && (
            <ComponentProperties
              schema={selectedSchema}
              control={control}
              basePath={name}
            />
          )}
        </FormItem>
      )}
    />
  );
} 