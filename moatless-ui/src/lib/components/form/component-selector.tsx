import { useState, useEffect } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/lib/components/ui/select"
import { useComponents } from "@/lib/hooks/useFlowComponents"
import { Loader2, Info } from "lucide-react"
import { ComponentProperty } from "./component-property"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/lib/components/ui/card"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/lib/components/ui/tooltip"

interface ComponentSelectorProps {
  componentType: string // e.g., "selectors", "value-functions", "feedback-generators"
  value: any // Can be null, string, or object with *_class property
  onChange: (componentValue: any) => void
}

export function ComponentSelector({ componentType, value, onChange }: ComponentSelectorProps) {
  const { data: components, isLoading } = useComponents(componentType)

  // Get the class property name based on component type
  const getClassPropertyName = (): string => {
    switch (componentType) {
      case 'selectors': return 'selector_class';
      case 'value-functions': return 'value_function_class';
      case 'feedback-generators': return 'feedback_generator_class';
      case 'artifact-handlers': return 'artifact_handler_class';
      case 'expanders': return 'expander_class';
      default: return `${componentType.replace(/-/g, '_')}_class`;
    }
  }

  const classPropertyName = getClassPropertyName();

  // Extract the selected component class from the value
  const getSelectedComponentClass = (): string => {
    if (!value) return "";
    if (typeof value === 'string') return value;
    if (typeof value === 'object' && value !== null) {
      return value[classPropertyName] || "";
    }
    return "";
  }

  const selectedComponentClass = getSelectedComponentClass();

  // Define a constant for the "None" option value
  const NONE_VALUE = "__none__"

  // Generate options from components data
  const options = components ?
    Object.keys(components).map(id => ({
      value: id,
      label: components[id].title || getComponentDisplayName(id),
      description: components[id].description || ''
    })) : []

  // Helper function to safely extract component name from ID
  const getComponentDisplayName = (id: string): string => {
    if (!id || typeof id !== 'string') return 'Unknown';
    const parts = id.split('.');
    return parts.length ? parts[parts.length - 1] : id;
  }

  // Get the selected component schema
  const selectedComponent = selectedComponentClass && components ? components[selectedComponentClass] : null

  // Handle component selection change
  const handleComponentChange = (newValue: string) => {
    if (newValue === NONE_VALUE) {
      // Handle "None" selection
      onChange(null);
    } else {
      // Create a new component object with the class property
      const newComponent = {
        [classPropertyName]: newValue
      };

      // Add default values from schema
      if (components && components[newValue]?.properties) {
        Object.entries(components[newValue].properties).forEach(([propName, propSchema]) => {
          // Skip type property and properties starting with underscore
          if (propName !== 'type' && !propName.startsWith('_') && propName !== classPropertyName) {
            // Use default from schema if available
            if ((propSchema as any).default !== undefined) {
              newComponent[propName] = (propSchema as any).default;
            }
          }
        });
      }

      onChange(newComponent);
    }
  }

  // Handle property change
  const handlePropertyChange = (propName: string, propValue: any) => {
    if (!value || typeof value !== 'object') return;

    const updatedComponent = {
      ...value,
      [propName]: propValue
    };

    onChange(updatedComponent);
  }

  // Get a friendly name for the component type
  const getComponentTypeName = () => {
    switch (componentType) {
      case 'selectors': return 'Selector';
      case 'value-functions': return 'Value Function';
      case 'feedback-generators': return 'Feedback Generator';
      case 'artifact-handlers': return 'Artifact Handler';
      default: return componentType.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    )
  }

  // Convert empty string value to NONE_VALUE for the Select component
  const selectValue = selectedComponentClass === "" ? NONE_VALUE : selectedComponentClass;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Select value={selectValue} onValueChange={handleComponentChange}>
          <SelectTrigger className="flex-1">
            <SelectValue placeholder={`Select ${getComponentTypeName()}`} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE_VALUE}>None</SelectItem>
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selectedComponent?.description && (
        <div className="mt-1">
          <p className="text-sm text-muted-foreground whitespace-pre-line">
            {selectedComponent.description}
          </p>
        </div>
      )}

      {selectedComponentClass && selectedComponent && Object.keys(selectedComponent.properties || {}).length > 0 && (
        <div className="space-y-4">
          {Object.entries(selectedComponent.properties || {}).map(([propName, propSchema]) => {
            // Skip type property, class property, and properties starting with underscore
            if (propName === 'type' || propName === classPropertyName || propName.startsWith('_')) return null;

            return (
              <ComponentProperty
                key={propName}
                id={propName}
                property={propSchema}
                value={value && typeof value === 'object' ? value[propName] : undefined}
                onChange={handlePropertyChange}
              />
            );
          })}
        </div>
      )}
    </div>
  )
}

